import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import json
import os
import time
from pynput.keyboard import Controller

class KeyboardMapper:
    def __init__(self):
        self.keyboard = Controller()
        self.input_file = "keyboard_input.txt"
        self.config_file = "key_mappings.json"
        self.presets_file = "key_presets.json"
        self.mappings = self.load_mappings()
        self.presets = self.load_presets()
        self.running = False
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Keyboard Mapper")
        self.root.geometry("550x600")
        self.root.resizable(True, True)
        
        # Create the UI
        self.create_ui()
        
        # Start monitoring for input
        self.start_monitoring()
        
    def load_mappings(self):
        # Default mappings
        default_mappings = {
            "0": "a",
            "1": "b",
            "2": "c",
            "3": "d",
            "4": "e",
            "5": "f",
            "6": "g",
            "7": "h",
            "8": "i",
            "9": "j",
            "10": "k",
            "11": "l",
            "12": "m",
            "13": "n"
        }
        
        # Try to load from file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Use defaults and create the file
        self.save_mappings(default_mappings)
        return default_mappings
    
    def load_presets(self):
        # Default presets
        default_presets = {
            "Default": {
                "0": "a",
                "1": "b",
                "2": "c",
                "3": "d",
                "4": "e",
                "5": "f",
                "6": "g",
                "7": "h",
                "8": "i",
                "9": "j",
                "10": "k",
                "11": "l",
                "12": "m",
                "13": "n"
            },
            "Arrow Keys": {
                "0": "up",
                "1": "down",
                "2": "left",
                "3": "right",
                "4": "space",
                "5": "enter",
                "6": "esc",
                "7": "tab",
                "8": "backspace",
                "9": "ctrl",
                "10": "shift",
                "11": "alt",
                "12": "page_up",
                "13": "page_down"
            }
        }
        
        # Try to load from file if it exists
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Use defaults and create the file
        self.save_presets(default_presets)
        return default_presets
    
    def save_mappings(self, mappings):
        with open(self.config_file, 'w') as f:
            json.dump(mappings, f, indent=4)
    
    def save_presets(self, presets):
        with open(self.presets_file, 'w') as f:
            json.dump(presets, f, indent=4)
    
    def create_ui(self):
        # Create a frame for the controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Start/Stop button
        self.toggle_button = ttk.Button(control_frame, text="Start", command=self.toggle_monitoring)
        self.toggle_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Stopped")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Save button
        save_button = ttk.Button(control_frame, text="Save Mappings", command=self.save_current_mappings)
        save_button.pack(side=tk.RIGHT, padx=5)
        
        # Create a frame for presets
        preset_frame = ttk.LabelFrame(self.root, text="Presets")
        preset_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create preset controls
        preset_controls = ttk.Frame(preset_frame)
        preset_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(preset_controls, text="Preset:").pack(side=tk.LEFT, padx=5)
        
        # Create a dropdown for presets
        self.preset_var = tk.StringVar()
        self.preset_dropdown = ttk.Combobox(preset_controls, textvariable=self.preset_var, state="readonly", width=20)
        self.preset_dropdown['values'] = list(self.presets.keys())
        self.preset_dropdown.current(0)  # Set to first preset
        self.preset_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Button to load a preset
        load_preset_btn = ttk.Button(preset_controls, text="Load", command=self.load_preset)
        load_preset_btn.pack(side=tk.LEFT, padx=5)
        
        # Button to save a preset
        save_preset_btn = ttk.Button(preset_controls, text="Save As New", command=self.save_as_preset)
        save_preset_btn.pack(side=tk.LEFT, padx=5)
        
        # Button to delete a preset
        delete_preset_btn = ttk.Button(preset_controls, text="Delete", command=self.delete_preset)
        delete_preset_btn.pack(side=tk.LEFT, padx=5)
        
        # Create a frame for the mapping entries
        mapping_frame = ttk.LabelFrame(self.root, text="Key Mappings")
        mapping_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a canvas with scrollbar for the mappings
        canvas = tk.Canvas(mapping_frame)
        scrollbar = ttk.Scrollbar(mapping_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create entry widgets for each mapping
        self.mapping_entries = {}
        for i in range(14):  # 0 to 13
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=5, pady=3)
            
            label = ttk.Label(frame, text=f"Input {i}:", width=10)
            label.pack(side=tk.LEFT)
            
            entry = ttk.Entry(frame, width=5)
            entry.insert(0, self.mappings.get(str(i), ""))
            entry.pack(side=tk.LEFT, padx=5)
            
            description = ttk.Label(frame, text="Enter a single character or special key name (e.g., 'space', 'enter')")
            description.pack(side=tk.LEFT, padx=5)
            
            self.mapping_entries[str(i)] = entry
        
        # Add log area
        log_frame = ttk.LabelFrame(self.root, text="Activity Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = tk.Text(log_frame, height=6)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        """ # Add a text widget to explain how to use the application
        help_frame = ttk.LabelFrame(self.root, text="Help")
        help_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        help_text = tk.Text(help_frame, height=4, wrap=tk.WORD)
        help_text.pack(fill=tk.BOTH)
        help_text.insert(tk.END, "This application reads numbers from another Python program via the 'keyboard_input.txt' file. " +
                         "When a number between 0-13 is received, the corresponding key will be pressed. " +
                         "You can customize which key is pressed for each input number and save presets for different applications.")
        help_text.config(state=tk.DISABLED) """
    
    def log(self, message):
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')}: {message}\n")
        self.log_text.see(tk.END)
    
    def save_current_mappings(self):
        new_mappings = {}
        for i in range(14):
            key = str(i)
            value = self.mapping_entries[key].get()
            if value:  # Only save non-empty values
                new_mappings[key] = value
        
        self.mappings = new_mappings
        self.save_mappings(new_mappings)
        self.log("Mappings saved successfully")
    
    def load_preset(self):
        preset_name = self.preset_var.get()
        if preset_name in self.presets:
            preset_mappings = self.presets[preset_name]
            # Update the UI
            for i in range(14):
                key = str(i)
                value = preset_mappings.get(key, "")
                self.mapping_entries[key].delete(0, tk.END)
                self.mapping_entries[key].insert(0, value)
            
            # Save as current mappings
            self.mappings = preset_mappings.copy()
            self.save_mappings(self.mappings)
            self.log(f"Loaded preset: {preset_name}")
        else:
            self.log(f"Preset not found: {preset_name}")
    
    def save_as_preset(self):
        # Get current mappings from UI
        new_mappings = {}
        for i in range(14):
            key = str(i)
            value = self.mapping_entries[key].get()
            if value:  # Only save non-empty values
                new_mappings[key] = value
        
        # Ask for preset name
        preset_name = simpledialog.askstring("Save Preset", "Enter a name for this preset:", parent=self.root)
        
        if preset_name:
            # Save the preset
            self.presets[preset_name] = new_mappings
            self.save_presets(self.presets)
            
            # Update the dropdown
            self.preset_dropdown['values'] = list(self.presets.keys())
            self.preset_var.set(preset_name)
            
            self.log(f"Saved preset: {preset_name}")
        else:
            self.log("Preset save cancelled")
    
    def delete_preset(self):
        preset_name = self.preset_var.get()
        if preset_name == "Default":
            messagebox.showwarning("Cannot Delete", "The Default preset cannot be deleted.")
            return
            
        if preset_name in self.presets:
            confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the preset '{preset_name}'?")
            if confirm:
                del self.presets[preset_name]
                self.save_presets(self.presets)
                
                # Update the dropdown
                self.preset_dropdown['values'] = list(self.presets.keys())
                self.preset_var.set("Default")  # Reset to Default
                
                self.log(f"Deleted preset: {preset_name}")
        else:
            self.log(f"Preset not found: {preset_name}")
    
    def toggle_monitoring(self):
        if self.running:
            self.running = False
            self.toggle_button.config(text="Start")
            self.status_label.config(text="Status: Stopped")
            self.log("Monitoring stopped")
        else:
            self.running = True
            self.toggle_button.config(text="Stop")
            self.status_label.config(text="Status: Running")
            self.log("Monitoring started")
    
    def start_monitoring(self):
        self.check_for_input()
        self.root.mainloop()
    
    def check_for_input(self):
        if self.running:
            if os.path.exists(self.input_file):
                try:
                    with open(self.input_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            self.process_input(content)
                            # Clear the file after reading
                            with open(self.input_file, 'w') as f:
                                f.write("")
                except Exception as e:
                    self.log(f"Error reading input: {str(e)}")
        
        # Schedule the next check
        self.root.after(100, self.check_for_input)
    
    def process_input(self, input_text):
        try:
            num = int(input_text)
            if 0 <= num <= 13:
                key = self.mappings.get(str(num), "")
                if key:
                    self.keyboard.press(key)
                    self.keyboard.release(key)
                    self.log(f"Pressed key '{key}' for input {num}")
                else:
                    self.log(f"No key mapping for input {num}")
            else:
                self.log(f"Input out of range (0-13): {num}")
        except ValueError:
            self.log(f"Invalid input: {input_text}")

if __name__ == "__main__":
    app = KeyboardMapper()