import threading
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import pygame
import serial


class App:
    def __init__(self, root):
        pygame.mixer.init()
        self.root = root
        self.root.title("ESP32 DJ")
        self.path = None
        self.serial = None
        self.playing = False
        tk.Button(root, text="Select Song", command=self.load_song).pack(fill="x")
        tk.Button(root, text="Play", command=self.play).pack(fill="x")
        tk.Button(root, text="Pause", command=self.pause).pack(fill="x")
        tk.Button(root, text="Connect ESP32", command=self.connect_serial).pack(fill="x")
        self.status = tk.Label(root, anchor="w")
        self.status.pack(fill="x")
        root.protocol("WM_DELETE_WINDOW", self.close)

    def load_song(self):
        path = filedialog.askopenfilename(filetypes=[("MP3", "*.mp3")])
        if path:
            self.path = path
            pygame.mixer.music.load(path)
            self.status.config(text=path)

    def play(self):
        if not self.path:
            self.load_song()
        if not self.path:
            return
        if self.playing:
            pygame.mixer.music.unpause()
        else:
            pygame.mixer.music.play()
        self.playing = True

    def pause(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
        self.playing = False

    def connect_serial(self):
        if self.serial and self.serial.is_open:
            return
        port = simpledialog.askstring("ESP32 Port", "Enter port (e.g. COM3 or /dev/ttyUSB0):")
        if not port:
            return
        try:
            self.serial = serial.Serial(port, 115200, timeout=1)
        except serial.SerialException as err:
            messagebox.showerror("Serial Error", str(err))
            self.serial = None
            return
        self.status.config(text=f"Serial connected: {port}")
        threading.Thread(target=self.read_serial, daemon=True).start()

    def read_serial(self):
        while self.serial and self.serial.is_open:
            try:
                cmd = self.serial.readline().decode(errors="ignore").strip().upper()
            except serial.SerialException:
                break
            if cmd:
                self.root.after(0, lambda c=cmd: self.handle_serial(c))
        self.root.after(0, lambda: self.status.config(text="Serial disconnected"))

    def handle_serial(self, cmd):
        if cmd == "PLAY":
            self.play()
        elif cmd == "PAUSE":
            self.pause()
        elif cmd == "TOGGLE":
            self.pause() if self.playing else self.play()

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
        pygame.mixer.music.stop()
        self.root.destroy()


if __name__ == "__main__":
    App(tk.Tk()).root.mainloop()


