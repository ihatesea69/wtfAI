import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def compare_results(pred_file, true_file):
    # Đọc các file CSV
    pred_df = pd.read_csv(pred_file)
    true_df = pd.read_csv(true_file)
    
    # Đảm bảo các file có cùng số lượng và thứ tự
    assert len(pred_df) == len(true_df), "Số lượng dự đoán không khớp!"
    assert all(pred_df['id'] == true_df['id']), "ID không khớp!"
    
    # Tính accuracy
    accuracy = accuracy_score(true_df['type'], pred_df['type'])
    print(f"\nĐộ chính xác tổng thể: {accuracy*100:.2f}%")
    
    # Tạo confusion matrix
    cm = confusion_matrix(true_df['type'], pred_df['type'])
    
    # Vẽ confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # In classification report
    print("\nBáo cáo chi tiết:")
    print(classification_report(true_df['type'], pred_df['type']))
    
    # So sánh chi tiết từng lớp
    print("\nSo sánh chi tiết:")
    for class_id in sorted(true_df['type'].unique()):
        true_count = sum(true_df['type'] == class_id)
        pred_count = sum(pred_df['type'] == class_id)
        correct = sum((true_df['type'] == class_id) & (pred_df['type'] == class_id))
        print(f"\nLớp {class_id}:")
        print(f"- Số lượng thực tế: {true_count}")
        print(f"- Số lượng dự đoán: {pred_count}")
        print(f"- Dự đoán đúng: {correct}")
        print(f"- Độ chính xác: {correct/true_count*100:.2f}%")
    
    # Tìm các vị trí dự đoán sai
    incorrect_mask = pred_df['type'] != true_df['type']
    incorrect_predictions = pd.DataFrame({
        'id': pred_df.loc[incorrect_mask, 'id'],
        'true_type': true_df.loc[incorrect_mask, 'type'],
        'predicted_type': pred_df.loc[incorrect_mask, 'type']
    })
    
    if len(incorrect_predictions) > 0:
        print("\nCác dự đoán sai:")
        print(incorrect_predictions)
        incorrect_predictions.to_csv('incorrect_predictions.csv', index=False)
        print("\nĐã lưu danh sách các dự đoán sai vào file 'incorrect_predictions.csv'")
    else:
        print("\nKhông có dự đoán sai!")

if __name__ == "__main__":
    # File kết quả của model
    submission_file = "submission.csv"  # File dự đoán của model
    true_file = "ketqua.csv"  # File ground truth
    
    try:
        compare_results(submission_file, true_file)
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file. {str(e)}")
    except Exception as e:
        print(f"Lỗi: {str(e)}") 