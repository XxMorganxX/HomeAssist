def get_emeet_device(): 
    import pyaudio
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels', 0) > 0:
            if "emeet" in info.get('name').lower():
                print(f"Found EMEET device at index #{i}")
                return i
            
    p.terminate()
