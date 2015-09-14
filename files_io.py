import sys
import codecs
import ast


def load_text(path):
    with codecs.open(path, 'r', 'utf-8') as in_file:
        text = in_file.read()
    return text


def serialize(obj, path):
	"""
	Serializing object using repr() function, so that later it is easily deserializable
	"""
	with codecs.open(path, 'w', 'utf-8') as out_file:
		out_file.write(repr(obj))


def deserialize(path):
	"""
	Deserializing list using ast.literal_eval() function
	"""
	with codecs.open(path, 'r', 'utf-8') as in_file:
		obj_str = in_file.read()
	#print(len(obj_str), "symbols deserialized")
	obj = ast.literal_eval(obj_str)
	#print("object of type {0} deserialized, length = {1}".format(type(obj), len(obj)))
	return obj
