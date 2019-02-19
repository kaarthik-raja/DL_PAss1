
import argparse as agp
def csv_list(string):
   return string.split(',')

def main():
	print("parsing...\n")
	parser = agp.ArgumentParser()
	parser.add_argument("--lr", type=float, help="the learning rate", default=0.01)
	parser.add_argument("--momemtum", type=float, help="the momemtum in lr", default=0.5)
	parser.add_argument("--num_hidden", type=int, help="# of Hidden Layers", default= 3)
	parser.add_argument("--sizes", type=csv_list, help="# of Hidden Layers", default= 3)
	args=parser.parse_args()
	print(args.sizes)
	#initwts()

if __name__=="__main__":
	main()