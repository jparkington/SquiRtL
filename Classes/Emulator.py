from numpy import array, array_equal
from pyboy import PyBoy

class Emulator:
    def __init__(self, rom_path):
        self.pyboy_instance = PyBoy(rom_path, window = "SDL2")
        self.rom_path       = rom_path

        self.pyboy_instance.set_emulation_speed(0)

    def check_event_flag(self, event_address, bit_position):
        event_value = self.pyboy_instance.memory[event_address]
        return (event_value >> bit_position) & 1 == 1

    def close_emulator(self):
        self.pyboy_instance.stop()

    def get_screen_data(self):
        return array(self.pyboy_instance.screen.ndarray, copy = True)

    def press_button(self, button, frames = 10):
        initial_state = self.get_screen_data()

        self.pyboy_instance.button(button, delay = frames)
        
        for _ in range(frames):
            self.pyboy_instance.tick()
        
        new_state = self.get_screen_data()
        is_effective = not array_equal(initial_state, new_state)
        return is_effective, new_state

    def reset_emulator(self):
        self.close_emulator()
        self.pyboy_instance = PyBoy(self.rom_path, window = "SDL2")
        return self.get_screen_data()