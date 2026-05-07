import datetime
import numpy as np
import os

class savedat:
    def __init__(self, filename="simulation.dat"):
        self.filename = filename
        mode = "a" if os.path.exists(filename) else "w"
        with open(self.filename, mode) as f:
            f.write("[HEADER]\n")

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Time: {current_time}\n")


    def write(self, data):
        with open(self.filename, "a") as f:
            if isinstance(data, (list, tuple)):
                for item in data:
                    f.write(f"{item}\n")
            else:
                f.write(f"{data}\n")
