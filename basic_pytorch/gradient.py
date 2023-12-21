##******************************************************************************
 #
 #     Copyright (c) 2023 Bin Tan
 #
 ##*****************************************************************************/
import torch

def main():
    t : torch.Tensor = torch.tensor([0, 1, 4, 9]);
    print(t.shape);
    print(t)
    dts = torch.gradient(t)
    dim_idx = 0
    for dt in dts:
        print("----- gradient along dim ", dim_idx, " ----")
        print (dt.shape)
        print (dt)
        dim_idx += 1

if __name__ == "__main__":
     main()

