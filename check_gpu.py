import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("\nGPU Details:")
print(tf.config.list_physical_devices('GPU'))

# Kiểm tra chi tiết
if tf.test.is_built_with_cuda():
    print("\nTensorFlow được build với CUDA")
else:
    print("\nTensorFlow KHÔNG được build với CUDA")

# Thử một phép tính đơn giản trên GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print("\nKết quả phép nhân ma trận trên GPU:")
    print(c) 