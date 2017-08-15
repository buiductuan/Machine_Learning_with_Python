import mglearn as ml
import matplotlib.pyplot as plt

#lấy 1 điểm láng giềng gần nhất
# ml.plots.plot_knn_classification(n_neighbors=1)

#Lấy 3 điểm láng giềng gần nhất
ml.plots.plot_knn_classification(n_neighbors=3)

#đặt tiêu đề cho ảnh
plt.title('forge_one_neighbor')

#hiển thị ảnh
plt.show()