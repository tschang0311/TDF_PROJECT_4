#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <Adafruit_seesaw.h>
#include <SPI.h>
#include <Wire.h>
#include <vector>

// Feather ESP32-S2 TFT display wiring (use board defaults when available)
#ifndef TFT_CS
#define TFT_CS 7
#endif
#ifndef TFT_DC
#define TFT_DC 39
#endif
#ifndef TFT_RST
#define TFT_RST 40
#endif
#ifndef TFT_BACKLIGHT
#define TFT_BACKLIGHT 45
#endif
#ifndef TFT_I2C_POWER
#define TFT_I2C_POWER 21
#endif

constexpr uint16_t SCREEN_WIDTH = 240;
constexpr uint16_t SCREEN_HEIGHT = 135;
constexpr uint16_t COLOR_DARKGREY = 0x39E7;  // RGB565 dark grey
constexpr uint16_t CONTENT_X = 8;
constexpr uint16_t CONTENT_Y = 0;
constexpr uint16_t CONTENT_WIDTH = SCREEN_WIDTH - (CONTENT_X * 2);
constexpr uint16_t CONTENT_HEIGHT = SCREEN_HEIGHT - CONTENT_Y - 16;
constexpr char TRACK_SEPARATOR = '\t';
constexpr uint8_t MAX_VISIBLE_TRACKS = 6;
constexpr uint8_t LINE_HEIGHT = 18;
constexpr uint8_t ENCODER_I2C_ADDRESS = 0x49;
constexpr int32_t ENCODER_COUNTS_PER_STEP = 4;  // ANO encoder emits 4 counts per detent
constexpr uint16_t ENCODER_PRODUCT_ID = 5740;
constexpr uint8_t SS_SWITCH_SELECT = 1;
constexpr uint8_t SS_SWITCH_UP = 2;
constexpr uint8_t SS_SWITCH_LEFT = 3;
constexpr uint8_t SS_SWITCH_DOWN = 4;
constexpr uint8_t SS_SWITCH_RIGHT = 5;

Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);
String currentPayload;
Adafruit_seesaw encoder;
bool encoderAvailable = false;
int32_t lastEncoderPosition = 0;
int32_t encoderRemainder = 0;
int32_t scrollOffset = 0;
int32_t focusedIndex = 0;
std::vector<String> currentTracks;
bool lastSelectButtonState = true;  // true = not pressed (pullup)
unsigned long lastButtonPressTime = 0;
const unsigned long BUTTON_DEBOUNCE_MS = 200;
std::vector<int32_t> playingIndices;  // Track which indices are currently playing
unsigned long lastFlashTime = 0;
const unsigned long FLASH_INTERVAL_MS = 1000;  // Flash once per second
bool flashState = false;  // Toggle for flashing animation

void drawChrome();
void renderTrackList(const std::vector<String> &tracks, uint16_t color = ST77XX_ORANGE, bool enumerate = true, int32_t startIndex = 0, int32_t focusIndex = -1, const std::vector<int32_t> &playingIndices = std::vector<int32_t>());
void renderMessage(const String &text, uint16_t color = COLOR_DARKGREY);
void handleSerialInput();
void updateEncoderScroll();
void handleButtonPresses();
void updateFlashAnimation();
std::vector<String> splitTracks(const String &text);
void resetScrollState();
void initEncoder();
bool verifyEncoderProduct();
void updateScrollOffsetFromFocus();
bool isPlaying(int32_t index);

void setup() {
  pinMode(TFT_BACKLIGHT, OUTPUT);
  digitalWrite(TFT_BACKLIGHT, HIGH);  // keep backlight on

  pinMode(TFT_I2C_POWER, OUTPUT);
  digitalWrite(TFT_I2C_POWER, HIGH);  // enable display power rail
  delay(10);                          // allow rail to stabilize

  Serial.begin(115200);
  while (!Serial && millis() < 5000) {
    delay(10);  // wait for native USB Serial
  }

  tft.init(SCREEN_HEIGHT, SCREEN_WIDTH);
  tft.setRotation(1);  // horizontal layout (240x135)
  tft.fillScreen(ST77XX_BLACK);

  Wire.begin();
  currentPayload = "";
  initEncoder();
  resetScrollState();
  drawChrome();
}

void loop() {
  handleSerialInput();
  updateEncoderScroll();
  handleButtonPresses();
  updateFlashAnimation();
}

void drawChrome() { renderMessage("Waiting for filenames...", COLOR_DARKGREY); }

void renderMessage(const String &text, uint16_t color) {
  tft.fillRect(CONTENT_X, CONTENT_Y, CONTENT_WIDTH, CONTENT_HEIGHT, ST77XX_BLACK);

  tft.setTextWrap(true);
  tft.setTextSize(2);
  tft.setTextColor(color);
  tft.setCursor(CONTENT_X, CONTENT_Y);
  tft.print(text);
}

void renderTrackList(const std::vector<String> &tracks, uint16_t color, bool enumerate, int32_t startIndex, int32_t focusIndex, const std::vector<int32_t> &playingIndices) {
  tft.fillRect(CONTENT_X, CONTENT_Y, CONTENT_WIDTH, CONTENT_HEIGHT, ST77XX_BLACK);
  if (tracks.empty()) {
    tft.setCursor(CONTENT_X, CONTENT_Y);
    tft.setTextWrap(false);
    tft.setTextSize(1);
    tft.setTextColor(COLOR_DARKGREY);
    tft.print("No tracks found");
    return;
  }

  const size_t total = tracks.size();
  const size_t maxStart = total > MAX_VISIBLE_TRACKS ? total - MAX_VISIBLE_TRACKS : 0;
  size_t safeStart = startIndex < 0 ? 0 : static_cast<size_t>(startIndex);
  if (safeStart > maxStart) {
    safeStart = maxStart;
  }

  tft.setTextWrap(false);
  tft.setTextSize(2);

  uint8_t rendered = 0;
  uint16_t cursorY = CONTENT_Y;
  size_t idx = safeStart;
  for (; idx < total && rendered < MAX_VISIBLE_TRACKS; ++idx) {
    bool isTrackPlaying = isPlaying(static_cast<int32_t>(idx));
    bool isTrackFocused = (focusIndex >= 0 && static_cast<size_t>(focusIndex) == idx);
    
    // Determine color: flashing for playing tracks, blue for focused, green for others
    uint16_t trackColor;
    if (isTrackPlaying && flashState) {
      // Flash: show bright color when flashState is true
      trackColor = ST77XX_YELLOW;
    } else if (isTrackPlaying && !flashState) {
      // Flash: show normal color when flashState is false
      trackColor = isTrackFocused ? ST77XX_BLUE : ST77XX_GREEN;
    } else {
      // Not playing: normal colors
      trackColor = isTrackFocused ? ST77XX_BLUE : ST77XX_GREEN;
    }
    
    tft.setTextColor(trackColor);
    tft.setCursor(CONTENT_X, cursorY);
    if (enumerate) {
      tft.print(String(idx + 1));
      tft.print(". ");
    }
    tft.print(tracks[idx]);
    cursorY += LINE_HEIGHT;
    rendered++;
  }

  if (idx < total) {
    tft.setTextSize(1);
    tft.setCursor(CONTENT_X, cursorY);
    tft.print("...");
  }
}

void handleSerialInput() {
  if (!Serial.available()) {
    return;
  }

  String incoming = Serial.readStringUntil('\n');
  incoming.trim();

  if (incoming.length() == 0) {
    return;
  }

  if (incoming.startsWith("@")) {
    return;  // ignore control tokens such as @outline.txt
  }

  // Handle status messages from Python
  if (incoming.startsWith("PLAYING:")) {
    int32_t index = incoming.substring(8).toInt();
    // Add to playing indices if not already there
    bool found = false;
    for (size_t i = 0; i < playingIndices.size(); i++) {
      if (playingIndices[i] == index) {
        found = true;
        break;
      }
    }
    if (!found) {
      playingIndices.push_back(index);
    }
    // Trigger a redraw to show the playing state
    if (!currentTracks.empty()) {
      updateScrollOffsetFromFocus();
      renderTrackList(currentTracks, ST77XX_GREEN, true, scrollOffset, focusedIndex, playingIndices);
    }
    return;
  }

  if (incoming.startsWith("STOPPED:")) {
    int32_t index = incoming.substring(8).toInt();
    // Remove from playing indices
    for (size_t i = 0; i < playingIndices.size(); i++) {
      if (playingIndices[i] == index) {
        playingIndices.erase(playingIndices.begin() + i);
        break;
      }
    }
    // Trigger a redraw to update the playing state
    if (!currentTracks.empty()) {
      updateScrollOffsetFromFocus();
      renderTrackList(currentTracks, ST77XX_GREEN, true, scrollOffset, focusedIndex, playingIndices);
    }
    return;
  }

  if (incoming == currentPayload) {
    return;  // no change
  }

  currentPayload = incoming;
  currentTracks = splitTracks(currentPayload);
  resetScrollState();
  playingIndices.clear();  // Clear playing indices when track list changes

  if (currentTracks.empty()) {
    renderMessage("No tracks found", COLOR_DARKGREY);
  } else {
    updateScrollOffsetFromFocus();
    renderTrackList(currentTracks, ST77XX_GREEN, true, scrollOffset, focusedIndex, playingIndices);
  }
}

void updateEncoderScroll() {
  if (!encoderAvailable || currentTracks.empty()) {
    return;
  }

  int32_t position = encoder.getEncoderPosition();
  if (position == lastEncoderPosition) {
    return;
  }

  int32_t rawDelta = position - lastEncoderPosition;
  lastEncoderPosition = position;
  encoderRemainder += rawDelta;

  int32_t detentDelta = encoderRemainder / ENCODER_COUNTS_PER_STEP;
  if (detentDelta == 0) {
    return;
  }

  // Move focused index by the detent delta (one line per detent)
  int32_t newFocusedIndex = focusedIndex + detentDelta;
  int32_t maxIndex = static_cast<int32_t>(currentTracks.size()) - 1;
  
  // Clamp focused index to valid range
  if (newFocusedIndex < 0) {
    newFocusedIndex = 0;
  } else if (newFocusedIndex > maxIndex) {
    newFocusedIndex = maxIndex;
  }

  // Only update if the focused index actually changed
  if (newFocusedIndex != focusedIndex) {
    focusedIndex = newFocusedIndex;
    updateScrollOffsetFromFocus();
    renderTrackList(currentTracks, ST77XX_GREEN, true, scrollOffset, focusedIndex, playingIndices);
    
    // Update encoder position to match focused index
    int32_t desiredEncoderPosition = focusedIndex * ENCODER_COUNTS_PER_STEP;
    encoder.setEncoderPosition(desiredEncoderPosition);
    lastEncoderPosition = desiredEncoderPosition;
  }

  encoderRemainder = 0;
}

void updateScrollOffsetFromFocus() {
  if (currentTracks.empty()) {
    scrollOffset = 0;
    return;
  }

  // Calculate scroll offset to keep focused item visible
  // Keep focused item at the top of the visible area
  int32_t maxOffset = currentTracks.size() > MAX_VISIBLE_TRACKS 
    ? static_cast<int32_t>(currentTracks.size() - MAX_VISIBLE_TRACKS) 
    : 0;
  
  // Set scroll offset to focused index, but clamp to maxOffset
  scrollOffset = focusedIndex;
  if (scrollOffset > maxOffset) {
    scrollOffset = maxOffset;
  }
  if (scrollOffset < 0) {
    scrollOffset = 0;
  }
}

std::vector<String> splitTracks(const String &text) {
  std::vector<String> entries;
  int start = 0;

  while (start < text.length()) {
    int separatorIndex = text.indexOf(TRACK_SEPARATOR, start);
    String entry = separatorIndex == -1 ? text.substring(start) : text.substring(start, separatorIndex);
    entry.trim();

    if (entry.length() > 0) {
      entries.push_back(entry);
    }

    if (separatorIndex == -1) {
      break;
    }
    start = separatorIndex + 1;
  }

  return entries;
}

void resetScrollState() {
  scrollOffset = 0;
  focusedIndex = 0;
  encoderRemainder = 0;
  if (encoderAvailable) {
    encoder.setEncoderPosition(0);
    lastEncoderPosition = 0;
  }
}

void initEncoder() {
  Serial.println("Looking for seesaw!");
  
  if (!encoder.begin(ENCODER_I2C_ADDRESS)) {
    Serial.println("Couldn't find seesaw on default address");
    Serial.println("Unable to find the seesaw encoder");
    return;
  }
  
  Serial.println("seesaw started");
  
  if (!verifyEncoderProduct()) {
    Serial.println("Wrong firmware loaded - encoder not compatible");
    return;
  }

  encoder.pinMode(SS_SWITCH_UP, INPUT_PULLUP);
  encoder.pinMode(SS_SWITCH_DOWN, INPUT_PULLUP);
  encoder.pinMode(SS_SWITCH_LEFT, INPUT_PULLUP);
  encoder.pinMode(SS_SWITCH_RIGHT, INPUT_PULLUP);
  encoder.pinMode(SS_SWITCH_SELECT, INPUT_PULLUP);

  // get starting position
  lastEncoderPosition = encoder.getEncoderPosition();

  Serial.println("Turning on interrupts");
  encoder.enableEncoderInterrupt();
  encoder.setGPIOInterrupts((uint32_t)1 << SS_SWITCH_UP, 1);
  
  encoderAvailable = true;
  Serial.println("ANO Rotary Encoder connected (1=yes,0=no): 1");
}

bool verifyEncoderProduct() {
  uint32_t versionRaw = encoder.getVersion();
  uint16_t productId = (versionRaw >> 16) & 0xFFFF;
  if (productId != ENCODER_PRODUCT_ID) {
    Serial.print("Wrong seesaw firmware? Expected product ");
    Serial.print(ENCODER_PRODUCT_ID);
    Serial.print(" but found ");
    Serial.println(productId);
    return false;
  }
  Serial.print("Found Product ");
  Serial.println(productId);
  return true;
}

bool isPlaying(int32_t index) {
  for (size_t i = 0; i < playingIndices.size(); i++) {
    if (playingIndices[i] == index) {
      return true;
    }
  }
  return false;
}

void updateFlashAnimation() {
  unsigned long currentTime = millis();
  
  // Check if it's time to toggle the flash state
  if (currentTime - lastFlashTime >= FLASH_INTERVAL_MS) {
    lastFlashTime = currentTime;
    flashState = !flashState;
    
    // Only redraw if there are playing tracks
    if (!playingIndices.empty() && !currentTracks.empty()) {
      updateScrollOffsetFromFocus();
      renderTrackList(currentTracks, ST77XX_GREEN, true, scrollOffset, focusedIndex, playingIndices);
    }
  }
}

void handleButtonPresses() {
  if (!encoderAvailable || currentTracks.empty()) {
    return;
  }

  bool currentSelectState = encoder.digitalRead(SS_SWITCH_SELECT);
  unsigned long currentTime = millis();

  // Check for button press (transition from HIGH to LOW due to pullup)
  if (lastSelectButtonState && !currentSelectState) {
    // Button was just pressed
    if (currentTime - lastButtonPressTime > BUTTON_DEBOUNCE_MS) {
      // Valid button press
      if (focusedIndex >= 0 && focusedIndex < static_cast<int32_t>(currentTracks.size())) {
        // If the focused track is currently playing, pause it
        if (isPlaying(focusedIndex)) {
          Serial.print("PAUSE:");
          Serial.println(focusedIndex);
        } else {
          // Otherwise, play the focused track
          Serial.print("PLAY:");
          Serial.println(focusedIndex);
        }
      }
      lastButtonPressTime = currentTime;
    }
  }

  lastSelectButtonState = currentSelectState;
}


