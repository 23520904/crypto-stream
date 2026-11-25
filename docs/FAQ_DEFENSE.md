# CẨM NANG VẤN ĐÁP BẢO VỆ ĐỒ ÁN BIG DATA (TOP 20 CÂU HỎI & TRẢ LỜI MẪU)
**Dành cho:** Nhóm 16 (Nguyễn Xuân Đình Lực, Nguyễn Hữu Thịnh, Huỳnh Đào Quốc Trọng)  
**Đề tài:** Hệ thống truy xuất dữ liệu Coinbase & dự đoán giá BTC/ETH thời gian thực (kiến trúc Lambda)  
**GVHD:** ThS. Hà Minh Tân

---

## PHẦN 1: KIẾN TRÚC & HẠ TẦNG BIG DATA (LAMBDA ARCHITECTURE)

### Câu 1: Tại sao nhóm lại chọn kiến trúc Lambda mà không dùng kiến trúc Kappa?
* **Trả lời:**
  * Kiến trúc Lambda tách biệt rõ ràng 2 tầng: **Batch Layer** (huấn luyện các mô hình học sâu phức tạp trên dữ liệu lịch sử nhiều năm) và **Speed Layer** (xử lý luồng tick thời gian thực bằng PySpark Streaming).
  * Trong bài toán tài chính, việc huấn luyện lại toàn bộ mạng nơ-ron BD-LSTM / Conv-LSTM ngay trên luồng streaming (Online Learning) tiềm ẩn rủi ro trôi mô hình (*Catastrophic Forgetting / Concept Drift*). Do đó, Lambda cho phép duy trì mô hình đã được kiểm định ổn định ở tầng Batch, đồng thời tầng Speed chỉ làm nhiệm vụ suy luận nhanh (Inference ~110ms), đảm bảo độ trễ thấp và độ tin cậy tuyệt đối.

### Câu 2: Trong Kafka, nhóm dùng cơ chế gì để đảm bảo không bị mất dữ liệu và không trùng lặp?
* **Trả lời:**
  * Nhóm cấu hình Producer với:
    1. `acks="all"`: Đảm bảo toàn bộ bản sao (in-sync replicas) đã ghi nhận bản tin trước khi báo thành công.
    2. `enable.idempotence=True`: Kích hoạt cơ chế Idempotent Producer để loại bỏ trùng lặp ở tầng mạng nếu có gói tin bị gửi lại.
  * Ngoài ra ở tầng ứng dụng, nhóm xây dựng cơ chế **Watchdog** và **đối chiếu `last_trade_id` trong Heartbeat**. Nếu phát hiện có khoảng trống giao dịch, tiến trình sẽ tự động gọi REST API để kéo bù (`source="backfill"`).

### Câu 3: Tại sao lại cần Watermark trong PySpark Structured Streaming? Nhóm đặt 15 giây dựa trên cơ sở nào?
* **Trả lời:**
  * Watermark là ngưỡng thời gian cho phép dữ liệu đến trễ (*Late Data*). Nếu không có watermark, Spark sẽ phải giữ trạng thái (*State*) trong RAM vô hạn để chờ dữ liệu cũ, dẫn đến tràn bộ nhớ (Out-Of-Memory).
  * Nhóm chọn **15 giây** vì độ trễ mạng từ Coinbase WebSocket về server thường dưới 1–2 giây; khoảng thời gian 15 giây là đủ rộng để đón nhận các tick bị delay trên đường truyền mà không làm phình to bộ nhớ đệm trạng thái của Spark.

### Câu 4: Nhóm khử trùng lặp dữ liệu trong Spark như thế nào khi kết hợp cả WebSocket và REST Backfill?
* **Trả lời:**
  * Cả tick trực tiếp (`source="ws"`) và tick tải bù (`source="backfill"`) đều có khóa định danh duy nhất là `product_id` và `trade_id`.
  * Nhóm sử dụng hàm: `dropDuplicatesWithinWatermark(["product_id", "trade_id"])`. Spark sẽ tự động tra cứu trong cửa sổ watermark 15s để loại bỏ bất kỳ bản tin nào có `trade_id` đã từng xuất hiện.

### Câu 5: Cơ chế ghi Parquet nguyên tử (Atomic Write) trong Speed Layer hoạt động ra sao và giải quyết vấn đề gì?
* **Trả lời:**
  * Khi Spark Streaming ghi file Parquet định kỳ mỗi phút, nếu Dashboard Streamlit đọc đồng thời trúng thời điểm file đang được ghi dở, Dashboard sẽ bị văng lỗi hỏng file (*Corrupt File Error*).
  * Nhóm giải quyết bằng cách: Ghi dữ liệu ra file tạm `_tmp-part-*.parquet`, sau khi ghi hoàn tất 100% thì gọi lệnh đổi tên nguyên tử `os.replace(tmp, official_path)`. Trên hệ điều hành Linux (POSIX), thao tác rename là nguyên tử (Atomic), đảm bảo dashboard luôn chỉ đọc file hoàn chỉnh.

---

## PHẦN 2: MÔ HÌNH HỌC MÁY & HỌC SÂU (MACHINE LEARNING)

### Câu 6: Tại sao RMSE của nhóm ở Batch M2 (~0.81 – 1.08) lại lớn hơn bài báo (~0.02 – 0.03)? Có phải mô hình bị lỗi không?
* **Trả lời (Câu hỏi điểm 10):**
  * Mô hình hoàn toàn chính xác. Sự chênh lệch này là do nhóm **tuân thủ nghiêm ngặt nguyên tắc Machine Learning: Chỉ fit bộ chuẩn hóa Min-Max trên tập Train** (`scaler_fit: train`) để chống rò rỉ dữ liệu (*Data Leakage*).
  * Trong giai đoạn Test (2020–2021), giá BTC bùng nổ vượt đỉnh lịch sử (\$60.000 so với đỉnh \$19.000 của tập train). Khi chuẩn hóa, các giá trị bị vượt ngưỡng 1.0 (đạt 2.5 – 3.0), khiến sai số RMSE trên thang chuẩn hóa tăng lên 0.81 – 1.08.
  * Bài báo gốc nhiều khả năng đã **fit Min-Max trên toàn bộ tập dữ liệu (cả train lẫn test)**, vô tình để lộ thông tin đỉnh giá tương lai vào tập huấn luyện, khiến RMSE của họ nhỏ li ti ở mức 0.02. Nếu nhóm đổi tham số sang `scaler_fit: all` thì kết quả cũng sẽ tiệm cận 0.02 như bài báo.

### Câu 7: Tại sao nhóm lại chọn mô hình deploy theo Validation Loss mà không chọn theo Test Loss?
* **Trả lời:**
  * Tập Test đóng vai trò là "đề thi thật" mô phỏng dữ liệu tương lai chưa từng xảy ra. Nếu nhóm lấy tập Test ra để chọn mô hình nào có điểm cao nhất đem đi deploy, tức là nhóm đã vi phạm nguyên tắc khoa học vì đã "nhìn trộm đề thi tương lai để chọn người đi thi" (*P-hacking / Overfitting on Test set*).
  * Do đó, nhóm chỉ căn cứ vào **Validation Loss (bài thi thử)** để chọn ra checkpoint tối ưu nhất đưa vào `artifacts/`, đảm bảo tính trung thực và khách quan khoa học.

### Câu 8: Conv-LSTM trong đồ án hoạt động như thế nào? Điểm khác biệt so với LSTM truyền thống là gì?
* **Trả lời:**
  * LSTM truyền thống chỉ dùng phép nhân ma trận (Dense/Linear) giữa vector đầu vào và vector trạng thái ẩn.
  * `ConvLSTM1D`: Thay thế toàn bộ các phép nhân ma trận ở các cổng (Input, Forget, Output gate) bằng **phép tích chập 1D (`Conv1d`)**.
  * Cửa sổ $W=6$ bước được chia thành 2 đoạn con ($S=2$), mỗi đoạn gồm 3 bước ($L=3$). Phép tích chập 1D giúp trích xuất các mẫu hình giá cục bộ (không gian giá), trong khi các cổng LSTM xâu chuỗi sự phụ thuộc theo thời gian.

### Câu 9: BD-LSTM (Bidirectional LSTM) có bị rò rỉ thông tin tương lai khi suy luận không?
* **Trả lời:**
  * Không. BD-LSTM chỉ chạy xuôi và chạy ngược **bên trong cửa sổ lịch sử đã qua** (từ $t-5$ đến $t$).
  * Chiều xuôi quét từ $t-5 \rightarrow t$, chiều ngược quét từ $t \rightarrow t-5$. Cả hai chiều đều chỉ nhìn dữ liệu đã xảy ra trong quá khứ, sau đó ghép 2 vector trạng thái ẩn cuối cùng lại để dự báo $t+1 \dots t+5$. Hoàn toàn không lấy dữ liệu của tương lai.

### Câu 10: Ý nghĩa và vai trò của biến `partner_close` trong chiến lược `multivariate*` là gì?
* **Trả lời:**
  * Khi dự báo BTC, `partner_close` là giá đóng cửa của ETH, và ngược lại.
  * Bitcoin và Ethereum chiếm tới 60–70% tổng vốn hóa thị trường tiền mã hóa và có hệ số tương quan rất cao ($r \approx 0.8$). Giữa hai coin này có **hiệu ứng dẫn dắt - độ trễ (Lead-Lag effect)** và **luân chuyển dòng tiền (Capital Rotation)**.
  * Việc bổ sung giá của đồng coin đối ứng đóng vai trò là một **biến ngoại sinh (Exogenous Feature)** cùng ngành, giúp mạng nơ-ron nhận diện sớm các tín hiệu thị trường trước khi đồng coin hiện tại kịp phản ứng.

---

## PHẦN 3: ĐÁNH GIÁ ONLINE & BENCHMARK THỰC NGHIỆM

### Câu 11: Tại sao ở bảng đánh giá Online (M4), mô hình Persistence lại có RMSE tốt hơn các mô hình Deep Learning?
* **Trả lời:**
  * Đây là hiện tượng **Chuyển miền dữ liệu (Domain Shift)**: Mô hình Deep Learning được huấn luyện trên dữ liệu nến ngày (daily), nhưng tầng Speed layer lại áp dụng trực tiếp lên nến 1 phút (1-minute).
  * Ở khung thời gian siêu ngắn 1–5 phút, biến động giá chịu chi phối lớn bởi nhiễu vi cấu trúc thị trường (*Microstructure Noise*) và bước đi ngẫu nhiên (*Random Walk*). Do đó, dự báo giữ nguyên giá nến liền trước (`persistence`) có sai số rất nhỏ. Nhóm đã ghi nhận và báo cáo trung thực hiện tượng này đúng với thực tế tài chính định lượng.

### Câu 12: Xu hướng sai số dự báo từ bước 1 đến bước 5 diễn ra như thế nào?
* **Trả lời:**
  * Sai số RMSE của tất cả các mô hình (kể cả baseline lẫn Deep Learning) đều **tăng dần đều từ bước $t+1$ đến bước $t+5$** (ví dụ Conv-LSTM từ 0.007 ở bước 1 tăng lên 0.022 ở bước 5).
  * Điều này hoàn toàn phù hợp với quy luật tự nhiên của chuỗi thời gian: Càng dự báo xa vào tương lai, độ bất định tích lũy càng lớn.

### Câu 13: Khi chạy Benchmark trên 1, 2 và 4 cores (M5), thông lượng và độ trễ thay đổi ra sao?
* **Trả lời:**
  * **Thông lượng (Throughput)**: Tăng từ **$5.021,9$ ticks/s (1 core) $\rightarrow$ $5.624,5$ ticks/s (2 cores, +12%) $\rightarrow$ $5.938,1$ ticks/s (4 cores, +18.2%)**.
  * **Độ trễ xử lý ($p50$)**: Giảm từ $59,0$ giây xuống $55,5$ giây khi có nhiều core hơn để giải phóng hàng đợi (backlog) nhanh hơn.
  * **Thời gian suy luận ($p50$)**: Duy trì ổn định ở mức **$\sim 110$ ms**, hoàn toàn đủ nhanh để chạy song song với chu kỳ nến 60 giây.

### Câu 14: Tại sao thông lượng từ 2 cores lên 4 cores không tăng gấp đôi (Linear Scalability 100%)?
* **Trả lời:**
  * Theo **Định luật Amdahl**, trong hệ thống phân tán luôn có phần xử lý tuần tự không thể song song hóa 100%:
    1. Quá trình serialize/deserialize dữ liệu JSON và nạp mạng Kafka.
    2. Chi phí đồng bộ hóa checkpoint và ghi Parquet xuống đĩa.
    3. Việc suy luận PyTorch diễn ra trên driver thread.
  * Do đó, mức tăng trưởng +18.2% là hoàn toàn hợp lý trong môi trường cụm máy cục bộ (*local cluster*).

---

## PHẦN 4: VẬN HÀNH 24/7 & BẢO MẬT HỆ THỐNG

### Câu 15: Nếu hệ thống mất điện hoặc server restart thì dữ liệu có bị mất không?
* **Trả lời:**
  * Không. Hệ thống có 3 lớp bảo vệ trạng thái:
    1. **Kafka Topic Retention**: Kafka lưu trữ toàn bộ tick trên đĩa cứng với cấu hình lưu trữ 60 ngày.
    2. **Spark Checkpointing**: Lưu lại chính xác offset cuối cùng đã đọc vào thư mục checkpoint. Khi bật lại, Spark đọc tiếp đúng từ offset đó mà không bị trùng hay sót.
    3. **Runtime Recovery**: File `streaming/runtime.py` có hàm `_recover()` tự động quét các file Parquet đã lưu trước đó để khôi phục lại lịch sử nến và phút dự báo cuối cùng.

### Câu 16: Nếu đồng coin vàng PAXG không có giao dịch trong nhiều phút thì hệ thống xử lý ra sao?
* **Trả lời:**
  * Trong `streaming/runtime.py`, hàm `_series()` và `_frame()` thực hiện tái lập chỉ mục thời gian liên tục (`reindex(freq="min")`) và áp dụng kỹ thuật **`ffill()` (Forward Fill)**: lấy giá đóng cửa gần nhất của PAXG điền vào các phút vắng bóng giao dịch. Nếu vẫn thiếu thì lấy giá vàng mặc định từ scaler. Điều này đảm bảo vector đặc trưng luôn đủ 6 chiều và không bao giờ bị NaN.

### Câu 17: Nhóm quản lý tài nguyên và chống tràn RAM khi chạy 24/7 như thế nào?
* **Trả lời:**
  1. Giới hạn lịch sử lưu trữ trong RAM: Cấu hình `streaming.keep_minutes: 600` (chỉ giữ nến 10 giờ gần nhất trong RAM).
  2. Dọn dẹp trạng thái Spark định kỳ qua Watermark (xóa bỏ dữ liệu cũ hơn 15s).
  3. Garbage collection tự động trong Python và gọi `torch.no_grad()` khi suy luận để giải phóng đồ thị tính toán.

### Câu 18: Nhóm đã bảo mật hệ thống như thế nào trong môi trường Docker?
* **Trả lời:**
  * Tách biệt mạng nội bộ giữa Kafka và bên ngoài (dùng Listener INTERNAL:29092 và EXTERNAL:9092).
  * Đưa các thư mục runtime, checkpoint, cache vào `.gitignore` để tránh rò rỉ mã nhị phân và dữ liệu nhạy cảm lên kho mã nguồn công khai.

---

## PHẦN 5: PHÂN CÔNG & ĐÓNG GÓP CỦA THÀNH VIÊN NHÓM 16

### Câu 19: Các thành viên trong nhóm 16 đã phân chia công việc như thế nào?
* **Trả lời:**
  * **Nguyễn Xuân Đình Lực (Trưởng nhóm):** Thiết kế kiến trúc tổng thể Lambda; chịu trách nhiệm tầng Speed Layer (PySpark Structured Streaming), engine suy luận mô hình và bài đo Benchmark đa lõi M5.
  * **Nguyễn Hữu Thịnh:** Chịu trách nhiệm toàn bộ tầng Ingestion (Coinbase WebSocket, Kafka Producer, cơ chế Watchdog, bắt mất gói và tự động REST Backfill M1), quản trị Docker.
  * **Huỳnh Đào Quốc Trọng:** Chịu trách nhiệm tầng Batch Layer (huấn luyện 30 runs các mô hình BD-LSTM, Conv-LSTM M2), phát triển Dashboard tương tác Streamlit (M6) và module đánh giá trực tuyến Online Evaluation (M4).

### Câu 20: Điểm nổi bật nhất mà đồ án của nhóm đạt được so với các đề tài khác là gì?
* **Trả lời:**
  1. **Đầy đủ 100% 6/6 mục tiêu chuẩn** từ Ingestion bền bỉ $\ge 45$ ngày đến Dashboard tương tác đa chiều.
  2. **Thực nghiệm chuẩn mực khoa học**: Không gian lận dữ liệu, chạy 30 lần lặp lại tính khoảng tin cậy 95% ($t$-Student).
  3. **Khả năng chịu lỗi công nghiệp**: Tự phát hiện mất gói và tự tải bù qua REST API.
  4. **Đo đạc mở rộng thực tế**: Đo đạc thông lượng và độ trễ trên 1, 2, 4 cores với cụm Spark phân tán.
