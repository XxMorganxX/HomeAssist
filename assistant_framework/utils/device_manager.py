def get_emeet_device():
    import pyaudio
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0 and "emeet" in info.get('name', '').lower():
                print(f"Found EMEET device at index #{i}")
                return i
        return None
    except Exception as e:
        try:
            print(f"Error enumerating audio devices: {e}")
        except Exception:
            pass
        return None
    finally:
        try:
            p.terminate()
        except Exception:
            pass
