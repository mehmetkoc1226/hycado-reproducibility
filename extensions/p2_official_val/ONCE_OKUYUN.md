# P2 resmi VAL dogrulamasi — v2 kimlik kontrolu duzeltmesi

v1, `preflight.json` icindeki `official_val.source_manifest_sha256` alanini
`data/prepared/manifest.json` dosyasinin SHA-256 degeri sanarak duruyordu.
Bu alan aslinda onceki resmi VAL `protocol.json` dosyasinin SHA-256 degerini
(`bf406f50...`) tasiyor. v2, bu alanin gercekten o protokole ait oldugunu,
onceki protokolun bagladigi hazir veri manifestinin SHA-256 degerinin bagimsiz
25 veri/model denetimindeki deger (`e2b610a2...`) ile ayni oldugunu ve
mevcut manifest dosyasinin da bu hash ile eslestigini kontrol eder. Ayrica
548 VAL kaydini onceki resmi VAL protokolunun kayitlariyla karsilastirir.
Bu duzeltme deney ayarlarini, model kodunu ve metrik kodunu degistirmez.

05 gelistirme kosusunu yeniden yapmayin. v2 ZIP'ini yeni bir klasore acin,
v1 klasorundeki Python dosyalarini degistirmeyin; yeni klasordeki
`06_P2_RESMI_VAL.cmd` dosyasini calistirin. Iki Python dosyasi ile CMD ayni klasorde
kalmalidir. Projeye dosya kopyalamayin ve eski tahminleri silmeyin.

Bu kosu onceden dondurulan bes yontemi 548 resmi VisDrone VAL goruntusunde
calistirir. Once her yontemden 3 goruntu kontrol edilir, sonra tam kosuya
gecilir. Her goruntu tahmini ayri dosyaya imzalanir; bilgisayar kapanirsa ayni
CMD kayitlari dogrulayarak devam eder. Butun tahminler bitene kadar VAL
annotation dosyalari acilmaz. YOLO 800 tahminleri ve metrikleri daha onceki
dondurulmus resmi VAL temel cizgisiyle eslesmelidir. Eslesmezse durur.

Rapor `D:\HyCADO_Temiz_Baslangic\experiments\p2_strong_baselines_v1_seed42\06_p2_resmi_val_rapor.zip`
konumundadir. Bu raporu gonderin; `official_val_predictions` klasoru ise
yerinde kalmalidir. Bu asama yeni egitim, parametre aramasi veya kontrollu
hiz/FPS olcumu yapmaz. Python metrikleri referans-alignedir, COCO mAP degildir.
