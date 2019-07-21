from load_model import *
import tensorflow as tf
import numpy as np


def load_model():
	with tf.gfile.GFile('model//./digital_gesture.pb', "rb") as f:  #读取模型数据
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read()) #得到模型中的计算图和数据
	with tf.Graph().as_default() as graph:  # 这里的Graph()要有括号，不然会报TypeError
		tf.import_graph_def(graph_def, name="")  # 导入模型中的图到现在这个新的计算图中，不指定名字的话默认是 import
		#for op in graph.get_operations():  # 打印出图中的节点信息
			#print(op.name, op.values())
	return graph

def predict(graph):
	im = Image.open("new_picture//./7.jpg")
	oldmat = np.asarray(im.convert('RGB'))
	with tf.Session() as sess:
		image_float = tf.image.convert_image_dtype(im, tf.float32)
		resized = tf.image.resize_images(image_float, [64, 64], method=3)
		resized_im = resized.eval()
		mat = np.asarray(resized_im).reshape(1, 64, 64, 3)
	# keep_prob = graph.get_tensor_by_name("keep_prob:0")
	x = graph.get_tensor_by_name("input_x:0")
	outlayer = graph.get_tensor_by_name("outlayer:0")
	prob = graph.get_tensor_by_name("probability:0")
	predict = graph.get_tensor_by_name("predict:0")

	with tf.Session(graph=graph) as sess:
		# print(sess.run(output))
		np.set_printoptions(suppress=True)
		out, prob, pred = sess.run([outlayer, prob, predict],feed_dict={x:mat})
		print(prob)
		display_result(oldmat,pred)

def display_result(mat, prediction):
	im = Image.fromarray(mat)#convert matrix to mat
	draw = ImageDraw.Draw(im)
	font = ImageFont.truetype("Deng.ttf", 150)
	draw.text((100, 100), "识别结果: {}".format(str(prediction)), fill= '#FF0000', font=font)
	im.show()

if __name__=="__main__":
	graph = load_model()
	predict(graph)