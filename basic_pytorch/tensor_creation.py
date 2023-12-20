##******************************************************************************
 #
 #     Copyright (c) 2023 Bin Tan
 #
 ##*****************************************************************************/
import torch


def main():
    # create a tensor from a Python list
    t1 : torch.Tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    print("---------- t1:")
    print(t1)

    t2 : torch.Tensor = torch.zeros(2, 3, 4)
    print("---------- t2:")
    print(t2)

    t3 : torch.Tensor = torch.full((2, 3, 4), fill_value=2.0)
    print("---------- t3:")
    print(t3)

    t4 : torch.Tensor = torch.rand([2, 3, 4])
    print("---------- t4:")
    print(t4)

    print(t4.shape)
    s1 : torch.Size = t4.size()
    print(s1)

    tt4 : torch.Tensor = t4[0]
    print("---------- t4 slice tt4 = t4[0]:")
    print(tt4)
    print(tt4.shape)

    ttt4 : torch.Tensor = t4[:,:,0];
    print("---------- t4 slice ttt4 = t4[:,:,0]:")
    print(ttt4)
    print(ttt4.shape)

    tt4[0,0] = 10.0
    print("---------- t4 element access:")
    print(t4[0,0,0])


    print("---------- t4 stride:")
    print(t4.stride())
    t5 : torch.Tensor = t4.view(2, 12)
    print("---------- t5 = t4.view(2,12):")
    print(t5.shape)
    print(t5.stride())
    print("Is t5 contiguous? ", t5.is_contiguous())
    print("Do t4 and t5 share the same underlying data? ",
          t4.storage().data_ptr() == t5.storage().data_ptr())
    print(t5)

    print("\n---------- view, transpose and reshape ---------:\n")
    t6 : torch.Tensor = t4.view(2,4,3)
    print("---------- t6 = t4.view(2,4,3):")
    print(t6.shape)
    print(t6.stride())
    print("Is t6 contiguous? ", t6.is_contiguous())
    print("Do t4 and t6 share the same underlying data? ",
          t4.storage().data_ptr() == t6.storage().data_ptr())
    print(t6)

    t7 : torch.Tensor = t4.transpose(1,2)
    print("---------- t7 = t4.transpose(1,2):")
    print(t7.shape)
    print(t7.stride())
    print("Is t7 contiguous? ", t7.is_contiguous())
    print("Do t4 and t7 share the same underlying data? ",
          t4.storage().data_ptr() == t7.storage().data_ptr())
    print(t7)

    t8 : torch.Tensor = t4.reshape((2,4,3))
    print("---------- t8 = t4.reshape((2,4,3)):")
    print(t8.shape)
    print(t8.stride())
    print("Is t8 contiguous? ", t8.is_contiguous())
    print("Do t4 and t8 share the same underlying data? ",
          t4.storage().data_ptr() == t8.storage().data_ptr())
    print(t8)


if __name__ == "__main__":
     main()
