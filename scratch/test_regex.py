import re

lines = [
    "13/10/2022 16.25 - Pesan dan panggilan terenkripsi secara end-to-end.",
    "13/10/2022 16.25 - Anda membuat grup",
    "13/10/2022 16.27 - amanda bergabung menggunakan tautan undangan grup ini",
    "13/10/2022 16.31 - Ajii: ca ini ajik",
    "12/11/24, 7:19 pm - Ajii: ca ini ajik",
    "12/11/24, 7:19 pm - Ajii: nak connect linkedln yak?"
]

pola_timestamp = re.compile(r'^\d{2}/\d{2}/\d{2,4}[, ]+\d{1,2}[:.]\d{2}(?:\s*[a-zA-Z]{2})? - ')
pola_chat = re.compile(r'^\d{2}/\d{2}/\d{2,4}[, ]+\d{1,2}[:.]\d{2}(?:\s*[a-zA-Z]{2})? - (.*?): (.*)')

print("Detection:")
print(any(pola_timestamp.match(sample_line) for sample_line in lines[:5]))

print("\nParsing:")
for line in lines:
    if pola_timestamp.match(line):
        cocok = pola_chat.match(line)
        if cocok:
            print(f"SENDER: {cocok.group(1)} | MSG: {cocok.group(2)}")
        else:
            print(f"SYSTEM: {line}")
    else:
        print(f"NOT WA: {line}")
