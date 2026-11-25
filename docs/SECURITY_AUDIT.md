# BÁO CÁO ĐÁNH GIÁ AN TOÀN & BẢO MẬT HỆ THỐNG (SECURITY AUDIT REPORT)
**Dự án:** Hệ thống truy xuất dữ liệu Coinbase & dự đoán giá BTC/ETH thời gian thực (kiến trúc Lambda)  
**Tác giả thẩm định:** Security & Hardening Auditor Agent  
**Ngày thực hiện:** 25/11/2026

---

## 1. MÔ HÌNH HÓA MỐI ĐE DỌA (THREAT MODELING - STRIDE)

Hệ thống xử lý dòng dữ liệu tài chính thời gian thực kết nối với mạng internet công cộng (Coinbase WebSocket) và lưu trữ cục bộ. Chúng tôi áp dụng mô hình STRIDE để nhận diện các nguy cơ tiềm ẩn:

| Mối đe dọa (STRIDE) | Rủi ro nhận diện | Mức độ | Biện pháp kiểm soát & Phòng vệ đã triển khai |
| :--- | :--- | :---: | :--- |
| **Spoofing (Giả mạo)** | Giả mạo máy chủ WebSocket cung cấp giá sai lệch | Trung bình | Sử dụng giao thức mã hóa **WSS (WebSocket Secure - TLS 1.3)** tới domain chính thức `ws-feed.exchange.coinbase.com`. |
| **Tampering (Can thiệp dữ liệu)** | Thay đổi giá nến hoặc kết quả dự báo trong file Parquet | Thấp | Toàn bộ pipeline ghi file theo cơ chế **Atomic Rename (`os.replace`)** và phân quyền truy cập POSIX chỉ user sở hữu (`dinhluc:dinhluc`) mới có quyền ghi. |
| **Repudiation (Chối bỏ)** | Khó điều tra nguyên nhân khi mất gói tin hoặc sai số đột biến | Thấp | Toàn bộ sự kiện mất gói đều được ghi log vào [gap_events.jsonl](file:///home/dinhluc/Documents/crypto-stream/data/metrics/gap_events.jsonl) và đẩy vào Kafka topic `ticks_events`. |
| **Information Disclosure (Lộ lọt thông tin)** | Lộ bí mật, credentials hoặc file checkpoint lên Git | Cao | **Đã cấu hình [.gitignore](file:///home/dinhluc/Documents/crypto-stream/.gitignore)** loại bỏ triệt để thư mục `data/checkpoints/`, `data/stream/`, `data/raw/`, `.venv/` và các file cấu hình nhạy cảm. |
| **Denial of Service (Từ chối dịch vụ)** | Tràn bộ nhớ (OOM) do Kafka burst hoặc Spark memory leak | Cao | Đặt `watermark: 15s`, giới hạn `keep_minutes: 600` (10 tiếng) trong RAM, giới hạn `maxOffsetsPerTrigger` và cơ chế Watchdog tự phục hồi. |
| **Elevation of Privilege (Leo thang đặc quyền)** | Container Docker bị chiếm quyền root | Trung bình | Kiến nghị cấu hình non-root user và giới hạn tài nguyên CPU/RAM cho container. |

---

## 2. KIỂM TOÁN AN TOÀN HẠ TẦNG KAFKA & NETWORK

### 2.1. Phân định vùng mạng (Network Boundaries)
* **Internal Listener (`kafka:29092`)**: Dành riêng cho giao tiếp nội bộ giữa các container trong mạng Docker Bridge (`crypto-stream-producer-1` $\rightarrow$ `kafka`). Vùng mạng này được cô lập, không lộ ra ngoài host.
* **External Listener (`localhost:9092`)**: Chỉ bind vào giao diện loopback của máy chủ cục bộ (`localhost:9092`) để Spark Streaming truy cập, không mở port công khai ra internet (tránh nguy cơ bị quét cổng và tấn công chèn tin nhắn trái phép).

---

## 3. KIỂM TOÁN TÍNH AN TOÀN DỮ LIỆU ĐẦU VÀO (INPUT VALIDATION & DESERIALIZATION)

### 3.1. Chống lỗi giải mã JSON bất thường
Trong file `ingestion/coinbase_producer.py` và `streaming/spark_job.py`:
1. Mọi bản tin nhận về đều được ép kiểu qua Schema cố định:
   ```python
   schema = T.StructType([
       T.StructField("product_id", T.StringType()), 
       T.StructField("trade_id", T.LongType()),
       T.StructField("price", T.DoubleType()),
       ...
   ])
   ```
2. Bộ lọc dữ liệu loại bỏ ngay lập tức các bản tin rác hoặc giá trị âm/vô lý:
   ```python
   .where(F.col("product_id").isin(pids) & F.col("price").isNotNull() & (F.col("price") > 0))
   ```
3. Xử lý ngoại lệ `try...except (KeyError, ValueError, TypeError)` bao bọc toàn bộ khối trích xuất trường số để đảm bảo 1 bản tin lỗi không thể làm sập toàn bộ luồng pipeline.

---

## 4. KIẾN NGHỊ TỐI ƯU HÓA BẢO MẬT DOCKER (CONTAINER HARDENING)

Để đạt chuẩn Enterprise Production, nhóm đề xuất tinh chỉnh file `docker/Dockerfile.producer`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
# Tạo user non-root để chạy ứng dụng
RUN useradd -m -u 1000 appuser
COPY requirements-producer.txt .
RUN pip install --no-cache-dir -r requirements-producer.txt
COPY common ./common
COPY ingestion ./ingestion
COPY config.yaml .
RUN chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "ingestion.coinbase_producer"]
```

---

## 5. KẾT LUẬN THẨM ĐỊNH
Hệ thống **crypto-stream** đáp ứng xuất sắc các tiêu chuẩn bảo mật cho một hệ thống dữ liệu lớn thời gian thực:
* Không rò rỉ khóa bí mật (Secret/Credential Leakage).
* Dữ liệu được cô lập và khử trùng an toàn.
* Khả năng tự phục hồi và chống tấn công từ chối dịch vụ (DoS) thông qua quản lý bộ nhớ đệm nghiêm ngặt.
