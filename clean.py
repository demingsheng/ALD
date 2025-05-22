import os
import json

def clear_all_files(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".png"):
                os.remove(os.path.join(root, file))
            if file.endswith(".xlsx"):
                os.remove(os.path.join(root, file))
            if file.endswith(".csv"):
                os.remove(os.path.join(root, file))
            if file.endswith(".out"):
                os.remove(os.path.join(root, file))
            if file.endswith(".json"):
                os.remove(os.path.join(root, file))
            if file.endswith(".pt"):
                os.remove(os.path.join(root, file))
            if file.endswith(".pkl"):
                os.remove(os.path.join(root, file))

            
if __name__ == "__main__":
    folder_to_clear = "./"  
    clear_all_files(folder_to_clear)
    print("Done!")