**1\. Mục tiêu nghiên cứu**  
Mục tiêu của đề tài là phát triển một phiên bản mở rộng của AA-CLIP theo hướng giảm phụ thuộc vào nhãn phân đoạn mức pixel trong quá trình huấn luyện.  
Cụ thể, đề tài hướng đến các mục tiêu sau:

* Giảm hoặc loại bỏ việc sử dụng ground-truth segmentation masks trong huấn luyện, vì đây là loại annotation có chi phí cao trong thực tế.    
* Giữ lại ý tưởng anomaly-aware text anchors của AA-CLIP, tức là vẫn học được sự tách biệt rõ ràng giữa đặc trưng “normal” và “anomaly” trong text space.   
* Duy trì khả năng định vị vùng bất thường ở mức patch/pixel dù không dùng mask thật trong train time. AA-CLIP gốc căn chỉnh patch-level visual features với text anchors để làm localization, do đó đây là phần cần được kế thừa và cải tiến. 

Mục tiêu chính của đề tài là:  
Xây dựng một biến thể của AA-CLIP theo hướng weakly supervised hoặc mask-free, nhằm giảm chi phí annotation nhưng vẫn bảo toàn hiệu quả phát hiện và định vị bất thường trong thiết lập zero-shot.  
**2\. Phương pháp nghiên cứu**  
Phương pháp đề xuất được xây dựng dựa trên khung hai giai đoạn của AA-CLIP. Trong paper gốc, Stage 1 sử dụng residual adapters ở text encoder để học các anomaly-aware text anchors, còn Stage 2 sử dụng residual adapters ở visual encoder để căn chỉnh patch-level visual features với các anchors này. Việc huấn luyện được thực hiện với classification loss và segmentation loss, trong đó segmentation loss gồm Dice loss và Focal loss dựa trên ground-truth mask, ngoài ra Stage 1 còn sử dụng Disentangle Loss để tăng độ tách biệt giữa normal anchor và anomaly anchor.  
Đề tài sẽ thay đổi ở cách supervision: thay vì phụ thuộc hoàn toàn vào ground-truth segmentation masks, mô hình sẽ sử dụng pseudo masks sinh từ patch-text similarity maps kết hợp với consistency learning và confidence weighting để huấn luyện localization theo hướng supervision yếu hơn  
2.1. Giữ lại anomaly-aware text anchors  
Trong giai đoạn đầu, mô hình vẫn học tách biệt giữa ngữ nghĩa “normal” và “anomaly” trong text space. Đây là phần cốt lõi của AA-CLIP vì paper cho thấy CLIP gốc có hiện tượng Anomaly Unawareness, tức là chưa phân biệt rõ giữa prompt mô tả bình thường và bất thường. AA-CLIP giải quyết bằng cách tạo ra các text anchors phân biệt hơn và bổ sung thêm Disentangle Loss để giảm tương quan giữa normal anchor và anomaly anchor.   
Trong đề tài này, phần học text anchors vẫn được giữ, nhưng quá trình huấn luyện sẽ hạn chế tối đa việc dùng mask phân đoạn thật.  
2.2. Thay segmentation mask bằng pseudo mask  
Thay vì dùng ground-truth pixel masks, mô hình sẽ khai thác patch-text similarity maps để sinh ra pseudo masks. Ý tưởng là:

* Từ visual features và anomaly-aware text anchors, mô hình tạo ra similarity map biểu diễn mức độ bất thường của từng patch. Cơ chế patch-text alignment này chính là một thành phần có trong AA-CLIP.   
* Từ similarity map đó, tiến hành chọn các vùng có độ tin cậy cao để tạo pseudo labels cho vùng bất thường.  
* Dùng pseudo masks này thay cho mask thật trong quá trình huấn luyện localization.

Để nâng cao chất lượng pseudo labels, đề tài sử dụng threshold theo từng ảnh thay vì một threshold cố định cho toàn bộ batch. Ví dụ, chỉ giữ lại top-k% patches có anomaly score cao nhất để tạo pseudo vùng bất thường. Sau đó, pseudo mask có thể được tinh lọc thêm bằng các bước xử lý như morphological refinement nhằm giảm nhiễu và làm mịn vùng bất thường.   
Như vậy, mô hình vẫn học được khả năng định vị anomaly nhưng không còn phụ thuộc hoàn toàn vào annotation mức pixel.  
2.3. Bổ sung consistency learning  
Vì pseudo masks thường có nhiễu, đề tài đề xuất thêm consistency loss để tăng độ ổn định. Chẳng hạn:

* ép các prediction maps ở nhiều mức đặc trưng khác nhau phải nhất quán với nhau  
* hoặc ép prediction map trước và sau augmentation nhẹ phải tương đồng.

Mục tiêu của phần này là giảm ảnh hưởng của pseudo labels nhiễu và giúp patch-level alignment học ổn định hơn, trong khi vẫn không yêu cầu phải có ground-truth mask hoàn chỉnh cho mọi ảnh 2.4. Huấn luyện weakly supervised  
2.4. Confidence weighting  
Không phải mọi pseudo mask đều đáng tin cậy như nhau. Vì vậy, đề tài bổ sung cơ chế confidence weighting để chỉ tăng cường học từ các ảnh có pseudo labels chất lượng cao. C  
Nếu anomaly map của một ảnh có độ tương phản rõ giữa vùng bất thường và nền, ảnh đó sẽ được sử dụng đầy đủ trong segmentation loss, nếu prediction map quá mơ hồ hoặc phân mảnh, ảnh đó sẽ được gán trọng số nhỏ hơn hoặc chỉ đóng góp vào classification loss ở mức ảnh. Giúp tránh việc mô hình học lại lỗi từ các pseudo labels kém chất lượng  
Khác với AA-CLIP gốc, mô hình đề xuất sẽ ưu tiên dùng: image-level anomaly labels, pseudo masks sinh từ similarity maps, consistency loss và confidence weighting, thay cho việc phụ thuộc hoàn toàn vào ground-truth segmentation masks.  
2.5. Hàm loss đề xuất  
Trong AA-CLIP gốc, loss tổng thể bao gồm:  
BCE classification loss với nhãn ảnh, Dice loss \+ Focal loss với ground-truth mask, và Disentangle Loss trong text space.   
Đề tài đề xuất thay phần segmentation supervision bằng pseudo supervision, cụ thể:  
Phương pháp đề xuất:  
Pseudo mask → weighted pseudo segmentation loss  
\+ consistency loss  
\+ confidence weighting  
\+ classification loss  
\+ disentangle loss  
**3\. Kế hoạch thực hiện**  
Kế hoạch thực hiện đề tài được chia thành 4 giai đoạn chính.  
Giai đoạn 1: Khảo sát và phân tích paper

* Đọc kỹ paper AA-CLIP để hiểu rõ pipeline 2 giai đoạn, các thành phần residual adapters, text anchors, patch alignment và các hàm loss được sử dụng. Paper mô tả Stage 1 là “Disentangling Anomaly-Aware Text Anchors” và Stage 2 là “Aligning Patch Features According to Text Anchors”.   
* Phân tích điểm mạnh và hạn chế của phương pháp, đặc biệt là sự phụ thuộc vào classification labels và segmentation masks trong huấn luyện. 

Giai đoạn 2: Xây dựng baseline

* Tái hiện hoặc chạy lại mô hình AA-CLIP gốc làm baseline.  
* Kiểm tra các kết quả cơ bản trên một số dataset tiêu biểu như MVTec-AD hoặc VisA (hai benchmark quan trọng trong thực nghiệm của paper).   
* Xác định điểm chuẩn về:   
  * image-level AUROC  
  * pixel-level AUROC. 

Giai đoạn 3: Phát triển phương pháp đề xuất

* Thiết kế cơ chế sinh pseudo masks từ patch-text similarity maps.  
* Bổ sung consistency loss và confidence weighting để giảm nhiễu từ pseudo labels.  
* Xây dựng phiên bản weakly supervised / mask-free của AA-CLIP.  
* So sánh các biến thể:   
  * dùng ground-truth mask đầy đủ  
  * dùng image-level label \+ pseudo mask  
  * dùng image-level label \+ pseudo mask \+ consistency loss  
  * dùng image-level label \+ pseudo mask \+ consistency loss \+ confidence weighting.

Giai đoạn 4: Thực nghiệm và đánh giá

* So sánh phương pháp đề xuất với AA-CLIP gốc và các baseline liên quan.  
* Đánh giá trên thiết lập zero-shot như paper gốc.   
* Phân tích:   
  * hiệu quả định vị bất thường,  
  * khả năng tổng quát hóa,  
  * độ chênh lệch khi bỏ ground-truth masks.  
* Thực hiện ablation study để chứng minh vai trò của:   
* pseudo masks  
* consistency loss  
* confidence weighting  
* anomaly-aware text anchors. 

**4\. Kỳ vọng kết quả**  
Đề tài kỳ vọng đạt được các kết quả sau:  
4.1. Về mô hình  
Xây dựng được một phiên bản weakly supervised / mask-free của AA-CLIP vẫn giữ được kiến trúc cốt lõi gồm:

* anomaly-aware text adaptation,  
* patch-level visual alignment,  
* residual adapter-based controlled adaptation.

4.2. Về hiệu quả  
Mô hình đề xuất được kỳ vọng:

* giảm đáng kể nhu cầu dùng pixel-level segmentation masks trong train time;  
* vẫn đạt hiệu quả anomaly detection đủ tốt ở cả mức ảnh và mức vùng;  
* duy trì được khả năng zero-shot ở mức cạnh tranh với AA-CLIP gốc. 

4.3. Về ý nghĩa nghiên cứu  
Đề tài sẽ chỉ ra rằng: hiệu quả của AA-CLIP không chỉ đến từ segmentation supervision mạnh, mà còn đến từ chính cơ chế anomaly-aware text anchors và cách chúng dẫn hướng cho visual features.   
Nếu kết quả khả quan, đề tài có thể chứng minh rằng zero-shot anomaly detection không nhất thiết phải phụ thuộc nhiều vào pixel-level masks như trong AA-CLIP gốc, từ đó mở ra hướng nghiên cứu thực tế hơn cho anomaly detection trong môi trường thiếu annotation.  
4.4. Về ứng dụng  
Đề tài có tiềm năng ứng dụng trong:

* kiểm tra lỗi công nghiệp, nơi việc gán nhãn vùng lỗi bằng tay rất tốn thời gian  
* y tế, nơi annotation mức pixel cần chuyên gia và có chi phí rất cao.

Do đó, một mô hình giảm phụ thuộc vào mask sẽ có giá trị thực tiễn cao hơn.