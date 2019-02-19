
import argparse as agp
def csv_list(string):
   return [ int(i) for i in string.split(',')]

def annealf(string):
    if string in ["true","True","T","t","1" ] :
        return True
    elif string in ["False","false","F","f","0"]:
        return False

def main():
    print("parsing...")
    parser = agp.ArgumentParser()
    parser.add_argument("--lr", type=float, help="the learning rate", default=0.01)
    parser.add_argument("--momemtum", type=float, help="the momemtum in lr", default=0.5)
    parser.add_argument("--num_hidden", type=int, help="# of Hidden Layers", default= 3)
    parser.add_argument("--sizes", type=csv_list, help="# of Nodes per H_Layer", default= [100,100,100])
    parser.add_argument("--activation", type=str, help="activation function", default= "sigmoid", choices=["sigmoid","tanh"])
    parser.add_argument("--loss", type=str, help="loss function", default= "ce", choices=["sq","ce"])
    parser.add_argument("--opt", type=str, help="optimizer", default= "gd", choices=["gd","momemtum","nag","adam"])
    parser.add_argument("--batch_size", type=int, help="batch size per step", default= 20)
    parser.add_argument("--epoch", type=int, help="# of EPOCHs", default= 5)
    parser.add_argument("--anneal", type=annealf, help="anneal", default= True,choices=[True,False])
    parser.add_argument("--save_dir", type=str, help="Save dir location", default= "pa1")
    parser.add_argument("--expt_dir", type=str, help="expt_dir location", default= "pa1")
    args=parser.parse_args()
    print(args.anneal)
    #initwts()

if __name__=="__main__":
    main()