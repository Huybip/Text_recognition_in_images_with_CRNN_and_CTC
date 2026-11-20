# Nhận dạng chữ trong ảnh bằng CRNN + BiLSTM + CTC

Đồ án nhận dạng chữ một dòng trong ảnh (tiếng Việt có dấu và tiếng Anh) bằng mạng **CRNN** tự xây dựng bằng PyTorch, huấn luyện với hàm mất mát **CTC**, chạy hoàn toàn trên **Google Colab**.

- Toàn bộ dự án nằm trong **một notebook**: `CRNN_OCR_HoanChinh_GiaiDoan1-7.ipynb`.
- Không dùng mô hình OCR pretrained. CNN, BiLSTM, CTC, hai bộ giải mã (Greedy, Beam Search) và các hàm CER đều được cài đặt trong notebook.
- Mọi số liệu đánh giá do notebook tính từ tập test khi bạn chạy. README này **không chứa kết quả**, xem mục [Kết quả](#kết-quả).

## Mục lục

1. [Bài toán và kiến trúc](#bài-toán-và-kiến-trúc)
2. [Cách chạy nhanh](#cách-chạy-nhanh)
3. [Các giai đoạn trong notebook](#các-giai-đoạn-trong-notebook)
4. [Dataset](#dataset)
5. [Mô hình và huấn luyện](#mô-hình-và-huấn-luyện)
6. [Đánh giá](#đánh-giá)
7. [Cấu trúc thư mục trên Google Drive](#cấu-trúc-thư-mục-trên-google-drive)
8. [Kết quả](#kết-quả)
9. [Lỗi thường gặp](#lỗi-thường-gặp)
10. [Hạn chế và hướng phát triển](#hạn-chế-và-hướng-phát-triển)
11. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

## Bài toán và kiến trúc

**Đầu vào:** ảnh chứa **một dòng** chữ. **Đầu ra:** chuỗi ký tự tương ứng.

````
Ảnh (1 x 48 x W)
   |
   v
CNN: 7 lớp Conv + BatchNorm + ReLU, MaxPool
   |   (B, 512, 1, W/2)
   v
Chuyển thành chuỗi: (T, B, 512), T = W/2
   |
   v
BiLSTM 2 tầng, hidden 256 mỗi chiều, dropout 0,3  ->  (T, B, 512)
   |
   v
Linear(512 -> NUM_CLASSES)   ->  (T, B, NUM_CLASSES)
   |
   v
CTC Loss (huấn luyện)  /  Greedy hoặc Beam Search (suy luận)
````

| Thành phần | Vai trò |
|---|---|
| CNN | Trích đặc trưng hình ảnh; giảm chiều cao về 1, giữ chiều rộng làm trục thời gian |
| BiLSTM | Học ngữ cảnh theo chuỗi, đọc cả hai chiều trái-phải và phải-trái |
| CTC | Học chuyển chuỗi `T` bước thành nhãn `L` ký tự mà không cần căn chỉnh từng ký tự; ký tự BLANK ngăn gộp nhầm các ký tự lặp |

Vocabulary gồm **214 ký tự** (chữ không dấu, chữ tiếng Việt có dấu hoa và thường, chữ số, dấu cách, dấu câu). Chỉ số 0 dành cho **BLANK**, nên `NUM_CLASSES = 215`.

## Cách chạy nhanh

Yêu cầu: tài khoản Google, Google Colab (GPU T4 miễn phí là đủ). Không cần cài gì trên máy cá nhân.

1. Tải file `CRNN_OCR_HoanChinh_GiaiDoan1-7.ipynb` lên Colab (File → Upload notebook).
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Ở **Cell 1.3**, đặt `MODE`:
   - `"small"`: chạy thử (2.000 / 400 / 400 ảnh train / val / test), 40 epoch.
   - `"full"`: kết quả tốt hơn (15.000 / 2.500 / 2.500 ảnh), 30 epoch.
4. Chạy **Run all**. Lần đầu Colab sẽ xin quyền truy cập Google Drive.
5. Khi chạy xong, kết quả nằm trên Drive (xem [cấu trúc thư mục](#cấu-trúc-thư-mục-trên-google-drive)).

**Chạy lại về sau:** mở lại notebook và Run all. Dataset được khôi phục từ Drive. Nếu checkpoint đã hoàn tất, bước huấn luyện tự bỏ qua.

**Khuyến nghị:** chạy `small` trước để kiểm tra pipeline. Khi val CER đã giảm rõ thì mới chuyển sang `full`.

## Các giai đoạn trong notebook

| Giai đoạn | Cell | Nội dung |
|---|---|---|
| 1 | 1.1 – 1.10 | Môi trường, sinh dataset, kiểm tra chất lượng, trực quan hóa |
| 2 | 2.0 – 2.2 | Cấu hình huấn luyện, tiền xử lý, augmentation |
| 3 | 3.1 – 3.3 | Vocabulary, mô hình CRNN, kiểm tra kích thước tensor |
| 4 | 4.1 – 4.3 | CTC Loss, kiểm tra điều kiện CTC, Greedy và Beam Search, unit test |
| 5 | 5.1 – 5.3 | Hàm CER, vòng huấn luyện (checkpoint, resume), đường cong loss |
| 6 | 6.1 – 6.4 | Đánh giá trên test, kiểm thử độ bền, upload ảnh |
| 7 | 7.1 – 7.6 | Phân tích lỗi, báo cáo tự động, cách dùng lại mô hình |

## Dataset

Dữ liệu được **tự sinh bằng Pillow** (không dùng TextRecognitionDataGenerator).

**Vì sao không dùng TRDG:** thư viện này đã ngừng bảo trì và phụ thuộc phiên bản Pillow cũ, dễ xung đột với môi trường Colab hiện tại. Bộ sinh tự viết kiểm soát tốt hơn font Unicode và augmentation.

**Nội dung văn bản** (nhiều nhóm để phủ đủ ký tự):

- Cụm từ và từ đơn tiếng Việt có dấu (viết thường, viết hoa chữ đầu, viết hoa toàn bộ).
- Cụm từ và từ đơn tiếng Anh (khoảng 280 từ, có đủ các chữ `w f j z q k x`).
- Chữ in hoa: viết tắt, dãy bảng chữ cái (`ABCDEFG`), chuỗi chữ hoa lẫn số.
- Số, ngày tháng, điện thoại, tiền tệ, phần trăm, chuỗi chữ lẫn số.
- Nhóm nguyên âm tiếng Việt đủ 5 thanh điệu.
- Chuỗi ký tự dễ nhầm (`0 O 1 l I 5 S`) và chuỗi ký tự ngẫu nhiên phủ toàn bộ vocabulary.

**Điều kiện ảnh:** nhiều font (chỉ giữ font có đủ glyph tiếng Việt, kiểm tra bằng `fontTools`), cỡ chữ 28–64, màu chữ và nền có kiểm tra tương phản, nghiêng ±3°, làm mờ, đổi sáng và tương phản, nhiễu Gaussian.

**Chống rò rỉ dữ liệu:** các tập được chia theo **nội dung văn bản**. Mỗi nội dung chỉ xuất hiện ở đúng một tập, kể cả khi chỉ khác hoa/thường hoặc dấu câu cuối. Notebook kiểm tra bằng `assert`.

**Chuẩn hóa nhãn:** Unicode **NFC**.

**Cấu trúc:**

````
dataset_<MODE>_en/
    train/  images/  labels.csv
    val/    images/  labels.csv
    test/   images/  labels.csv
````

`labels.csv` có hai cột: `image_path` (đường dẫn tương đối so với thư mục gốc dataset) và `text`.

**Lưu ý về dữ liệu tổng hợp:** các từ trong câu `The quick brown fox jumps over the lazy dog` được cố ý loại khỏi kho từ tiếng Anh, để câu này dùng làm phép thử không bị "học thuộc".

## Mô hình và huấn luyện

| Mục | Giá trị |
|---|---|
| Ảnh vào | Xám, cao 48 px, giữ tỷ lệ, chuẩn hóa về `[-1, 1]`, padding bằng màu nền của chính ảnh |
| Số bước thời gian | `T = W / 2` |
| Tham số | khoảng 8,55 triệu |
| Optimizer | AdamW, `lr = 1e-3`, `weight_decay = 1e-2` |
| Scheduler | OneCycleLR |
| Batch size | 64 |
| Độ chính xác hỗn hợp | AMP trên GPU; loss CTC luôn tính ở float32 |
| Gradient clipping | 5,0 |
| Early stopping | Theo val CER (hòa thì xét val loss), `PATIENCE = 8`. Chỉ đếm khi CER đã dưới 100% |
| Augmentation | Chỉ ở tập train: nghiêng, xiên, sáng/tương phản, mờ, nhiễu, đổi lề. Không co giãn tỷ lệ |

**Về giai đoạn "toàn BLANK":** với CTC, những epoch đầu thường cho val CER đúng 100% (mô hình chỉ phát ra BLANK) trước khi bắt đầu học. Đây là hiện tượng bình thường, không phải lỗi. Với `MODE="small"` mình đặt 40 epoch (khoảng 1.250 bước tối ưu) để mô hình có đủ bước vượt qua giai đoạn này.

**Checkpoint** (lưu mỗi epoch vào Drive):

- `last.pt`: toàn bộ trạng thái (mô hình, optimizer, scheduler, scaler, lịch sử) để tiếp tục huấn luyện.
- `best.pt`: mô hình có val CER thấp nhất, dùng khi suy luận.

Muốn huấn luyện lại từ đầu với cấu hình khác: đổi `RUN_NAME` (Cell 2.0). Muốn thay đổi cách sinh dữ liệu: đổi `DATA_TAG` (Cell 1.3).

## Đánh giá

Đánh giá trên **tập test độc lập**. Văn bản tham chiếu và dự đoán được chuẩn hóa nhất quán trước khi so sánh (NFC, bỏ khoảng trắng thừa, phân biệt hoa/thường).

- **CER** = (S + D + I) / N, tính trên toàn tập test; khoảng trắng tính là ký tự. Hàm CER tự cài đặt, có phân tách S/D/I và được đối chiếu với `jiwer`.
- **Word Accuracy** = tỷ lệ mẫu mà **toàn bộ chuỗi** dự đoán khớp 100% với tham chiếu. Chỉ số này khác với `1 - WER`.
- **Thời gian dự đoán** = (thời gian forward của mô hình + thời gian giải mã) / số ảnh, đo khi chạy theo batch, không tính thời gian đọc ảnh.

**Bộ giải mã:**

- **Greedy:** `argmax` mỗi bước, gộp ký tự lặp liên tiếp **trước**, rồi mới bỏ BLANK.
- **Beam Search (prefix beam search):** giữ 10 tiền tố tốt nhất, cộng xác suất các đường đi dẫn tới cùng một chuỗi, cắt tỉa ký tự có xác suất dưới 0,001. Không dùng mô hình ngôn ngữ.

**Kiểm thử bổ sung:** độ bền theo điều kiện ảnh (tối, nhiễu, nghiêng, mờ), nhóm ảnh có ký tự dễ nhầm, nhóm ảnh nhiều dấu tiếng Việt, và chức năng upload ảnh của riêng bạn (Cell 6.4).

**Phân tích lỗi** (Cell 7.1 – 7.3) tự phân loại mẫu sai: dự đoán rỗng, nhầm hoa/thường, sai dấu/thanh tiếng Việt, nhầm chữ số và chữ cái, nhầm ký tự hình dạng gần giống, mất ký tự đầu/cuối, sai khoảng trắng, và nhóm khác.

## Cấu trúc thư mục trên Google Drive

````
MyDrive/OCR_CRNN/
├── charset.json                      # bộ ký tự dùng chung
├── dataset_<MODE>_en.zip             # dataset nén, khôi phục tự động khi chạy lại
├── BAO_CAO_crnn_<MODE>_en.md         # báo cáo thực nghiệm tự sinh (số liệu thật)
└── checkpoints/
    └── crnn_<MODE>_en/
        ├── best.pt                   # mô hình tốt nhất (theo val CER)
        ├── last.pt                   # trạng thái để resume
        ├── history.json              # loss, CER, lr, thời gian từng epoch
        ├── loss_curves.png           # đường cong Training Loss, Validation Loss, Validation CER
        ├── test_predictions.csv      # dự đoán Greedy và Beam trên tập test
        ├── test_results.json         # CER, Word Accuracy, thời gian
        └── error_table.csv           # bảng phân loại lỗi
````

## Kết quả

> **README này không chứa số liệu thực nghiệm.** Kết quả thật phụ thuộc vào lần chạy của bạn và được notebook tự tính, in ra ở Cell 6.2 và ghi vào `BAO_CAO_*.md`. Điền bảng dưới đây sau khi chạy xong.

| Phương pháp | CER (%) | Word Accuracy (%) | Thời gian dự đoán (ms/ảnh) |
|---|---:|---:|---:|
| CRNN + Greedy | _chưa chạy_ | _chưa chạy_ | _chưa chạy_ |
| CRNN + Beam Search (w=10) | _chưa chạy_ | _chưa chạy_ | _chưa chạy_ |

**Trạng thái kiểm chứng của notebook:**

- Đã chạy toàn bộ notebook trong môi trường thử (CPU, tập con nhỏ): luồng chạy đúng, unit test decoder và CTC đạt, CER tự cài đặt khớp `jiwer`.
- Đã xác nhận mô hình **học được** (CER giảm từ 100% xuống khoảng 53%) trong một bài thử overfit 24 ảnh. Đây là kiểm tra khả năng học, không phải kết quả thực nghiệm.
- **Chưa** được chạy trên GPU/Colab thật trong môi trường phát triển, và chưa huấn luyện đầy đủ trên bộ dữ liệu hiện tại.

## Lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| Val CER đứng ở 100% nhiều epoch đầu | Giai đoạn "toàn BLANK" của CTC, bình thường. Nếu đến epoch 20 CER vẫn trên 95%, kiểm tra lại learning rate và dữ liệu |
| Dự đoán chỉ ra 1–2 ký tự hoặc rỗng | Mô hình chưa huấn luyện đủ. Tăng số epoch, đổi `RUN_NAME`, hoặc dùng `MODE="full"` |
| Sửa cấu hình mà huấn luyện không chạy lại | Checkpoint cũ đã đánh dấu hoàn tất. Đổi `RUN_NAME` để huấn luyện lại |
| `Quá ít font có tiếng Việt` | Chạy lại Cell 1.2 (lệnh `apt-get` cài font có thể lỗi mạng tạm thời) |
| `Zip trên Drive bị thiếu file` | Xóa file zip trong Drive rồi chạy lại Cell 1.8 |
| Đọc sai các chữ tiếng Anh hoặc chữ in hoa | Do dữ liệu tổng hợp thiếu các nhóm đó. Bản notebook hiện tại đã bổ sung từ tiếng Anh và nhóm chữ hoa; huấn luyện lại bằng `DATA_TAG` mới |
| Ảnh nhiều dòng cho kết quả kém | Mô hình chỉ huấn luyện trên ảnh một dòng. Hãy cắt từng dòng trước |

## Hạn chế và hướng phát triển

**Hạn chế**

1. Dữ liệu **tổng hợp** với từ vựng giới hạn và font hệ thống; chưa phản ánh chữ viết tay hay ảnh chụp ngoài đời thực.
2. Chỉ nhận dạng **một dòng chữ**; cần bộ tách dòng cho ảnh tài liệu.
3. CTC giả định các bước độc lập có điều kiện và không có mô hình ngôn ngữ, nên khó phân biệt ký tự giống nhau khi thiếu ngữ cảnh.
4. Chiều cao 48 px có thể làm mất dấu rất nhỏ.
5. Beam Search không dùng mô hình ngôn ngữ nên lợi ích so với Greedy thường nhỏ.
6. Từ vựng dùng chung giữa các tập (chỉ nội dung văn bản là không trùng), nên tập test đo khả năng đọc câu mới chứ chưa đo từ vựng hoàn toàn mới.

**Hướng phát triển**

- Dùng văn bản thật (báo, sách, Wikipedia) để sinh dữ liệu và tăng lên 50.000 ảnh trở lên.
- Thêm mô hình ngôn ngữ hoặc từ điển vào Beam Search.
- Thử `IMG_H = 64`, backbone ResNet, hoặc attention thay BiLSTM.
- Thêm bước tách dòng cho ảnh tài liệu.
- Kiểm thử trên ảnh chụp bằng điện thoại và chữ viết tay.
- Đối chiếu với VietOCR hoặc TrOCR **trên đúng tập test của dự án**, báo cáo ở bảng riêng, tách khỏi kết quả CRNN tự xây dựng.

## Tài liệu tham khảo

- CRNN PyTorch: https://github.com/meijieru/crnn.pytorch
- PARSeq: https://github.com/baudm/parseq
- TrOCR (Hugging Face): https://huggingface.co/docs/transformers/model_doc/trocr
- VietOCR: https://github.com/pbcquoc/vietocr
- TextRecognitionDataGenerator: https://github.com/Belval/TextRecognitionDataGenerator
- Dataset tham khảo: IIIT5K, MJSynth

Các mô hình PARSeq, TrOCR, VietOCR chỉ để tham khảo kiến trúc. Dự án này **không** dùng kết quả benchmark của chúng làm kết quả của mình.
````
````

Mình dùng khối bốn dấu backtick bên ngoài vì README chứa các khối code ba dấu backtick bên trong. Bạn chỉ cần sao chép phần nằm giữa, bỏ dòng ````markdown ở đầu và dòng ```` ở cuối.

Khi huấn luyện xong, bạn nhớ điền bảng ở mục "Kết quả" bằng số thật từ Cell 6.2 và cập nhật mục "Trạng thái kiểm chứng".
