/*
  TECHNOLOGY DESING FOUNDATIONS - PROJECT 4
  THOMAS CHANG, KENYA FOSTER, BRYCE PARSONS
  11/11/25

  ESP32 I/O Device to Python Controller

  The following program has been cleaned and commented
  for readibility with the help of ChatGPT. Additionally,
  this program was vibe-coded with the help of ChatGPT.
  AI assistance can help us rapidly experiment and test
  our ideas.

  Behavior:
    - Reads a single button on a configurable pin.
    - Supports INPUT_PULLUP or INPUT modes.
    - Debounces the input and classifies presses as SHORT or LONG.
    - Emits one of the following lines on Serial:
        IO:BTN_A:PRESS
        IO:BTN_A:LONG

  JSON configuration (send a single line terminated by newline):
    {
      "cmd": "cfg",
      "device": "BTN_A",
      "pin": 13,
      "mode": "INPUT_PULLUP",                   // or "INPUT"
      "debounce_ms": 60,                        // legacy key (kept for compatibility)
      "long_ms": 500,                           // legacy key (kept for compatibility)
      "debounceMilliseconds": 60,               // new long-form key (optional)
      "longPressMilliseconds": 500              // new long-form key (optional)
    }

  Dependencies:
    - ArduinoJson (install via Library Manager)
*/

#include <Arduino.h>
#include <ArduinoJson.h>

// -------------------------- Defaults (overridable via JSON) --------------------------
static int       defaultPin                     = 12;      // Default GPIO pin used for the button
static bool      defaultUsePullUpResistor       = true;    // true → INPUT_PULLUP, false → INPUT
static uint16_t  defaultDebounceMilliseconds    = 60;      // Minimum stable time to accept a change
static uint16_t  defaultLongPressMilliseconds   = 500;     // Threshold to classify a long press

// -------------------------- Mutable runtime configuration ----------------------------
// These reflect the *current* configuration (start with defaults, can be changed via JSON).
static int       buttonPin                      = defaultPin;
static bool      usePullUpResistor              = defaultUsePullUpResistor;
static uint16_t  debounceMilliseconds           = defaultDebounceMilliseconds;
static uint16_t  longPressMilliseconds          = defaultLongPressMilliseconds;

// -------------------------- Button state tracking ------------------------------------
// We maintain both the last *raw reading* and the last *stable reading* to implement debouncing.
static bool      lastStablePressed              = true;     // With pull-up: HIGH means not pressed, so "true" here means "not pressed" at boot
static bool      lastRawPressed                 = true;     // Most recent instantaneous read (before debounce)
static uint32_t  lastEdgeChangeTimeMilliseconds = 0;        // When the last raw change was seen (used for debounce timing)
static uint32_t  pressStartTimeMilliseconds     = 0;        // When a confirmed press began
static bool      isPressedStable                = false;    // Whether we are currently in the pressed state (stable)

// -------------------------- Serial line buffer for JSON config -----------------------
static String    serialLineBuffer;                           // Accumulates incoming characters until newline

// -------------------------------------------------------------------------------------
// readIsPressed: returns the logical "pressed" state (true when physically pressed).
// With INPUT_PULLUP wiring, the button reads LOW when pressed; otherwise, it reads HIGH.
// -------------------------------------------------------------------------------------
bool readIsPressed() {
  int rawDigitalValue = digitalRead(buttonPin);
  // If using pull-up, the pressed state is LOW; otherwise, it is HIGH.
  bool isPhysicallyPressed = usePullUpResistor ? (rawDigitalValue == LOW) : (rawDigitalValue == HIGH);
  return isPhysicallyPressed;
}

// -------------------------------------------------------------------------------------
// applyPinModeForButton: applies the appropriate pinMode to the configured button pin.
// -------------------------------------------------------------------------------------
void applyPinModeForButton() {
  // Choose INPUT or INPUT_PULLUP based on configuration.
  if (usePullUpResistor) {
    pinMode(buttonPin, INPUT_PULLUP);
  } else {
    pinMode(buttonPin, INPUT);
  }
}

// -------------------------------------------------------------------------------------
// readUnsignedOrFallback<T>: helper to safely read an unsigned integer from JSON
// using either a primary key or a secondary fallback key.
// -------------------------------------------------------------------------------------
template <typename T>
T readUnsignedOrFallback(const JsonVariantConst& json,
                         const char* primaryKey,
                         const char* secondaryKey,
                         T fallback) {
  if (json.containsKey(primaryKey)) {
    return json[primaryKey].as<T>();
  }
  if (secondaryKey && json.containsKey(secondaryKey)) {
    return json[secondaryKey].as<T>();
  }
  return fallback;
}

// -------------------------------------------------------------------------------------
// handleConfigJson: parses a single JSON line and updates configuration if cmd == "cfg".
// Supports both legacy snake_case keys and new long-form camelCase keys.
// -------------------------------------------------------------------------------------
void handleConfigJson(const String& jsonLine) {
  // Create a JSON document with a size that can handle our small configuration payload.
  StaticJsonDocument<256> jsonDocument;

  // Attempt to deserialize the incoming JSON line.
  DeserializationError parseError = deserializeJson(jsonDocument, jsonLine);
  if (parseError) {
    // Parsing failed; we silently ignore malformed lines to avoid noisy logs.
    return;
  }

  // Command type must be "cfg" to apply configuration.
  const char* commandString = jsonDocument["cmd"];
  if (!commandString || String(commandString) != "cfg") {
    // Not a configuration command; ignore other messages.
    return;
  }

  // ---------------- Pin configuration ----------------
  // Accept "pin" if provided; otherwise, keep the current pin.
  if (jsonDocument.containsKey("pin")) {
    // Read the new pin number and assign it.
    buttonPin = jsonDocument["pin"].as<int>();
  }

  // ---------------- Mode configuration ----------------
  // Accept "mode": "INPUT" or "INPUT_PULLUP" (case-insensitive).
  if (jsonDocument.containsKey("mode")) {
    String modeString = jsonDocument["mode"].as<String>();
    modeString.toUpperCase(); // Normalize for comparison
    // If the word "PULLUP" appears, we assume the user wants INPUT_PULLUP.
    usePullUpResistor = (modeString.indexOf("PULLUP") >= 0);
  }

  // ---------------- Timing configuration ----------------
  // Support both legacy keys and new long-form keys for debounce and long-press thresholds.
  debounceMilliseconds = readUnsignedOrFallback<uint16_t>(
      jsonDocument, "debounceMilliseconds", "debounce_ms", debounceMilliseconds);

  longPressMilliseconds = readUnsignedOrFallback<uint16_t>(
      jsonDocument, "longPressMilliseconds", "long_ms", longPressMilliseconds);

  // After updating configuration, apply the correct pin mode immediately.
  applyPinModeForButton();

  // Emit a concise configuration summary for visibility during development and debugging.
  Serial.print("CFG: pin=");
  Serial.print(buttonPin);
  Serial.print(" mode=");
  Serial.print(usePullUpResistor ? "INPUT_PULLUP" : "INPUT");
  Serial.print(" debounceMilliseconds=");
  Serial.print(debounceMilliseconds);
  Serial.print(" longPressMilliseconds=");
  Serial.println(longPressMilliseconds);
}

// -------------------------------------------------------------------------------------
// setup: Arduino entry point for hardware initialization and initial state capture.
// -------------------------------------------------------------------------------------
void setup() {
  // Start the serial interface for both configuration and event emission.
  Serial.begin(115200);

  // Allow the serial interface a brief moment to stabilize.
  delay(50);

  // Apply the pin mode based on the current configuration.
  applyPinModeForButton();

  // Initialize debouncing state using the immediate reading.
  lastRawPressed                 = readIsPressed();   // Take a first raw reading
  lastStablePressed              = lastRawPressed;    // Treat the first reading as stable at boot
  isPressedStable                = lastStablePressed; // Mirror stable state into "pressed" tracking
  lastEdgeChangeTimeMilliseconds = millis();          // Mark the time of this initial state

  // Emit startup markers to signal readiness to any listening host.
  Serial.println("READY");
  Serial.println("IO:BTN_A:READY");  // Harmless line to observe connection on host side
}

// -------------------------------------------------------------------------------------
// loop: main execution cycle; processes serial config and runs debounce + classification.
// -------------------------------------------------------------------------------------
void loop() {
  // -------------------------------------------------------------------
  // 1) Ingest JSON configuration from Serial (newline-terminated lines)
  // -------------------------------------------------------------------
  while (Serial.available() > 0) {
    // Read one character at a time from the serial buffer.
    char incomingCharacter = static_cast<char>(Serial.read());

    // Newline or carriage return signifies the end of the configuration line.
    if (incomingCharacter == '\n' || incomingCharacter == '\r') {
      // If we have accumulated content, attempt to parse and apply it.
      if (serialLineBuffer.length() > 0) {
        handleConfigJson(serialLineBuffer);
        serialLineBuffer = "";  // Clear the buffer for the next line.
      }
    } else {
      // Accumulate characters into the line buffer while avoiding excessive growth.
      if (serialLineBuffer.length() < 240) {
        serialLineBuffer += incomingCharacter;
      }
      // If the buffer is already large, we silently drop extra characters
      // to prevent memory bloat and keep the device responsive.
    }
  }

  // -------------------------------------------------------------------
  // 2) Button debouncing and press classification
  // -------------------------------------------------------------------
  // Capture the instantaneous logical pressed state (not yet debounced).
  bool currentRawPressed = readIsPressed();

  // Record the current time for debounce and duration calculations.
  uint32_t currentTimeMilliseconds = millis();

  // If the instantaneous reading changed since the previous cycle, note the edge time.
  if (currentRawPressed != lastRawPressed) {
    lastRawPressed                  = currentRawPressed;           // Update the last raw sample
    lastEdgeChangeTimeMilliseconds  = currentTimeMilliseconds;     // Start debounce window
  }

  // If the input has remained unchanged longer than the debounce threshold,
  // then we accept the new state as "stable."
  bool hasDebounceWindowElapsed =
      (currentTimeMilliseconds - lastEdgeChangeTimeMilliseconds) >= debounceMilliseconds;

  // Only update the stable state if:
  //  - The debounce window has elapsed, and
  //  - The stable state differs from the current raw reading.
  if (hasDebounceWindowElapsed && (currentRawPressed != lastStablePressed)) {
    // Accept the new stable state after the debounce period.
    lastStablePressed = currentRawPressed;

    if (lastStablePressed == true) {
      // -------------------- Transition: RELEASED → PRESSED --------------------
      // We have just become pressed (stable). Begin timing this press.
      isPressedStable              = true;
      pressStartTimeMilliseconds   = currentTimeMilliseconds;

      // We do not emit an event yet; we wait until release to classify duration.
    } else {
      // -------------------- Transition: PRESSED → RELEASED --------------------
      // We have just become released (stable). If we were previously pressed,
      // compute the duration to distinguish between SHORT and LONG press types.
      if (isPressedStable) {
        uint32_t pressDurationMilliseconds = currentTimeMilliseconds - pressStartTimeMilliseconds;

        // Compare duration to the configured threshold.
        if (pressDurationMilliseconds >= longPressMilliseconds) {
          Serial.println("IO:BTN_A:LONG");   // Long press has been detected.
        } else {
          Serial.println("IO:BTN_A:PRESS");  // Short press has been detected.
        }
      }

      // We are no longer in the pressed state.
      isPressedStable = false;
    }
  }

  // -------------------------------------------------------------------
  // Optional behavior:
  // If you want to emit "LONG" while the button remains held (instead of on release),
  // you can enable this single-shot block. By default, we classify on release only.
  // -------------------------------------------------------------------
  /*
  if (isPressedStable && (currentTimeMilliseconds - pressStartTimeMilliseconds) >= longPressMilliseconds) {
    Serial.println("IO:BTN_A:LONG");  // Emit a long press once when threshold is crossed
    isPressedStable = false;          // Convert to single-shot to avoid repeated LONGs
  }
  */

  // Keep the loop lightweight to maintain responsive debouncing and serial handling.
  delay(1);
}
