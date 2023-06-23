import pandas as pd
from faker import Faker

# Rastgele veri seti oluşturma
n = 1000
fake = Faker()
isimler = [fake.name() for _ in range(n)]
cinsiyet = [fake.random_element(elements=('Erkek', 'Kadın')) for _ in range(n)]
yas = [fake.random_int(min=18, max=90) for _ in range(n)]
tansiyon = [fake.random_int(min=80, max=200) for _ in range(n)]
seker = [fake.random_element(elements=('Normal', 'Yüksek')) for _ in range(n)]
kolestrol = [fake.random_int(min=100, max=300) for _ in range(n)]
sigara = [fake.random_element(elements=('Hayır', 'Evet')) for _ in range(n)]
teşhis = [fake.random_element(elements=('Astım', 'Diabet', 'Hipertansiyon', 'Kanser', 'Kalp Hastalığı')) for _ in range(n)]

# Veri setini birleştirme
veri = pd.DataFrame({
    'Ad': isimler,
    'Cinsiyet': cinsiyet,
    'Yaş': yas,
    'Tansiyon': tansiyon,
    'Şeker': seker,
    'Kolestrol': kolestrol,
    'Sigara': sigara,
    'Teşhis': teşhis
})

# Veri setini CSV dosyasına yazma
veri.to_csv('hasta_verileri.csv', index=False)
