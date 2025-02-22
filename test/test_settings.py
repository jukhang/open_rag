from settings import settings

def test_settings():
    print(settings.model_dump())

if __name__ == "__main__":
    test_settings()