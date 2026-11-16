# Hazır HyCADO arşivlerini yayımlama

## Durum

Kod güncellemesi yayımlandı. Büyük arşivler henüz yayımlanmadı. 27. adım raporu sekiz ZIP dosyasının hazırlanıp doğrulandığını kaydediyor. Bunları yeniden paketlemeyin. Bu işlem eğitim, çıkarım veya Python paket kurulumu gerektirmez.

## Yapılacak işlem

1. Kendi bilgisayarınızda D:\HyCADO_Yayin_v1 klasörünü açın. Aşağıdaki sekiz ZIP ile release_manifest.json ve SHA256SUMS.txt dosyaları hazır olmalıdır.
2. https://github.com/mehmetkoc1226/hycado-reproducibility/releases sayfasını açın. Kendi GitHub hesabınızda oturum açıp **Draft a new release / Create a new release** düğmesine basın.
3. Tag alanına **v16-artifacts** yazıp yeni tag oluşturmayı seçin. Target olarak **main** seçin. Başlık: **HyCADO v16 checkpoints and frozen predictions**.
4. Açıklamaya bu repodaki docs/ARTIFACT_RELEASE_NOTES_v16.md dosyasının içeriğini kopyalayın. Yeni release formunun **Attach binaries** alanına aşağıdaki sekiz ZIP ile release_manifest.json ve SHA256SUMS.txt dosyalarını sürükleyin. Normal **Add file** sayfasına yüklemeyin. ZIP dosyalarını açmayın; birleştirmeyin.
5. Yükleme göstergelerinin tamamlanmasını bekleyin. Gerekirse **Save draft** ile kaydedin. Sekiz ZIP ve iki yardımcı dosyanın tamamı görünmeden yayımlamayın; yükleme kesilirse mevcut taslağı açıp eksik dosyaları ekleyin.
6. On dosya da göründüğünde **Publish release** düğmesine basın. Oluşan yayın bağlantısını sohbete gönderin. Ardından yayımlanan dosya adları, boyutları ve GitHub tarafından sunulan SHA256 değerleri kontrol edilecek; makale ve repo erişilebilirlik metni gerçek bağlantılarla güncellenecek.

## Sekiz büyük dosya

- `checkpoint_yolo.zip` (410,440,435 byte)
- `checkpoint_crop.zip` (526,970,188 byte)
- `checkpoint_full.zip` (526,962,508 byte)
- `predictions_stage18.zip` (318,739,034 byte)
- `predictions_stage19.zip` (43,315,801 byte)
- `predictions_stage20.zip` (807,481,946 byte)
- `predictions_stage24.zip` (97,534,553 byte)
- `legacy_replay_inputs.zip` (144,864,176 byte)

Ek iki dosya: `release_manifest.json`, `SHA256SUMS.txt`. Klasördeki eski `RELEASE_NOTES_EN.md` dosyasını eklemeniz gerekmez; yayın açıklamasında yukarıda belirtilen güncel v16 metnini kullanın. `27_yayin_paketleme_rapor.zip` dosyası bu yüklemenin parçası değildir.

Toplam ZIP boyutu yaklaşık 2,68 GiB. Yükleme süresi internetin gönderme hızına bağlıdır. Sadece bu reponun Releases sayfasında işlem yapın.

## Doğrulamanın kapsamı

Kodun 311bfe15c1a9df09326ec1d210b60c847905baed sürümü GitHub’dan temiz bir klasöre indirilerek denetlendi. 581 tarihsel dosya, 238 güncel manifest girdisi ve iki güncellenen doküman doğrulandı; yedi kök test geçti. Bu kontrol, büyük dosyaların yüklenmesini veya tüm eğitimin baştan tekrarını kapsamaz.

GitHub resmî yükleme adımları: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
