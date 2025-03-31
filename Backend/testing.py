import base64

# Your base64 string (Make sure it's a full string, not truncated)
base64_string = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAHqAAMDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD5/ooooAKKKKACiiigAopfy/KigDsvtLf3xRUXnp/z3X81ooAXZP8A3Iv/AB6ik3zeg/P/AOvRQAuJPV/z/wDrUU3B/vP/AN8CigBdzf8APSX/AMdopu+P/nsPy/8ArUUAP23H/PL9TRTNsnqn5/8A16KALWw/3l/75FFV/NP/ADzf9aKAND7Mv99/yorQwf8Anq/5f/XooAyftvuPyH+FFSfZbb/np+p/xooAqYb+4/8A3wf8KKgwv/PGP8hRQA7evov5H/CipN4/55n/AD+FFADdsn9z+f8AhRTfk/uyfmaKAJfM9xRTdw/uD8j/AIUUAL5f/TZ/z/8Ar0Ubf9pv+/QooAT5f+ey/mP8KKdsi/uD8v8A69FADtq/88o//Hf8KKZuX0f/AL+CigBMj1H/AHytFP3/APTS2/75FFADPMH92T/v63+NFGV/57N/3+/+vRQA37SP76fmaKk82T/nsP8Avof40UAM/e/894P8/wDAaKN4/wCe8f8A32v+NFADtyf30/76H+NFHl/9M3/8dooAbtP9x/8Ax+ipvLb+9L+Y/wDiqKAIt0HqP+/if/FUUvmJ7/kaKAG8+h/7+0VNsl/5+H/MUUAN3D+8P/Hf8KKPPk9W/wC+hRQBx+T2/nRQRk5HSigA57dPrRRuxwKKAFGMc0Uw9aKAP//Z"

# Ensure proper padding
missing_padding = len(base64_string) % 4
if missing_padding:
    base64_string += "=" * (4 - missing_padding)  # Fix padding

# Decode and save the image
try:
    image_data = base64.b64decode(base64_string, validate=True)
    with open("test_output.jpg", "wb") as f:
        f.write(image_data)
    print("✅ Saved test_output.jpg")
except Exception as e:
    print("❌ Error decoding Base64:", e)
