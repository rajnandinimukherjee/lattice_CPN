import ast
import argparse
import deformations as defs

def_names = ["HomogDef", "TorusDef", "ProjDef", "SkewMatDef", "FlowDef"]


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="CP(N) dimension", type=int, default=3)
    parser.add_argument("--latsize", help='lattice dimensions, eg "(64,64)"',
                        type=lambda s: tuple(ast.literal_eval(s)), default=(64, 64))
    parser.add_argument("--i", help="OnePointFn element i",
                        type=int, default=0)
    parser.add_argument("--j", help="OnePointFn element j",
                        type=int, default=0)
    parser.add_argument(
        "--beta", help="LatticeDdimAction parameter beta", type=float, default=4.0)
    parser.add_argument("--delta", help='two point function delta, eg "(0,1)"',
                        type=lambda s: tuple(ast.literal_eval(s)), default=(0, 1))
    parser.add_argument("--epochs", help="epochs", type=int, default=500)
    parser.add_argument("--batch_size", help="batch size",
                        type=int, default=1000)
    parser.add_argument("--split", help="training to test data split ratio",
                        type=float, default=0.8)
    parser.add_argument("--deformation", help=f"deformation model ({'|'.join(def_names)})",
                        type=str, default="HomogDef")
    parser.add_argument("--deftype", help=f"deformation type (constant|general)",
                        type=str, default="constant")
    parser.add_argument("--hidden_dim", help=f"parnet hidden layer dimension",
                        type=int, default=16)
    parser.add_argument("--run", help=f"run training", action="store_true")
    parser.add_argument(
        "--no-run", help=f"load samples only", action="store_false")
    parser.add_argument("--source", help=f"data source (Raj|Tej)",
                        type=str, default="Raj")
    parser.add_argument(
        "--update", help="print updates every 10 epochs", type=bool, default=False)
    args = parser.parse_args()

    assert args.deformation in def_names, f"Deformation type not recognized, choose from {
        ', '.join(def_names)}"

    if args.deformation == "TorusDef":
        deformation = defs.TorusDeformations
    elif args.deformation == "ProjDef":
        deformation = defs.ProjectorDeformations
    elif args.deformation == "SkewMatDef":
        deformation = defs.SkewMatrixDeformations
    elif args.deformation == "FlowDef":
        deformation = defs.FlowDeformations
    else:
        deformation = defs.HomogenousDeformations

    return args, deformation
