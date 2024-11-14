from guided_diffusion.image_datasets import AmbientDataset
import matplotlib.pyplot as plt
from guided_diffusion.fastmri_dataloader import (FastBrainMRI, uniformly_cartesian_mask, fmult, ftran, get_weighted_mask)


def main():
    dataset = AmbientDataset(
            "tst_large"
        )
    
    path = "/project/cigserver3/export1/g.harry/self-supervised-diffusion/test"
    for i in range(5):
        x, args = dataset[i]
        plt.imshow(abs(x), cmap='gray')
        plt.show()
        plt.savefig(f"{path}/x{i}")

        AtAx = args["AtAx"][0,:,:] + 1j * args["AtAx"][1,:,:]
        plt.imshow(abs(AtAx), cmap='gray')
        plt.show()
        plt.savefig(f"{path}/AtAx{i}")

        AtAhat_x = fmult(AtAx.unsqueeze(0), args["smps"], args["A"].unsqueeze(0))
        AtAhat_x = ftran(AtAhat_x, args["smps"], args["A"].unsqueeze(0))
        plt.imshow(abs(AtAhat_x[0,:,:]), cmap='gray')
        plt.show()
        plt.savefig(f"{path}/AtAAtAx{i}")

        AtAhat_x = fmult(x.unsqueeze(0), args["smps"], args["A_hat"].unsqueeze(0))
        AtAhat_x = ftran(AtAhat_x, args["smps"], args["A_hat"].unsqueeze(0))

        AtAAtAhat_x = fmult(AtAx.unsqueeze(0), args["smps"], args["A_hat"].unsqueeze(0))
        AtAAtAhat_x = ftran(AtAhat_x, args["smps"], args["A_hat"].unsqueeze(0))
        plt.imshow(abs(AtAAtAhat_x[0,:,:]), cmap='gray')
        plt.show()
        plt.savefig(f"{path}/AtAhat_x_fr{i}")
        
        print(args["A_hat"][0])
        print(args["A"][0])


    # print(args['x'].shape)
    # print(args['Ax'].shape)
    # print(args['mask'].shape)
    # print(im.shape)

    # plt.imshow(im[0,:,:], cmap='gray')
    # plt.show()
    # plt.savefig("x-hat")
    # plt.imshow(args["Ax"][0,:,:], cmap='gray')
    # plt.show()
    # plt.savefig("x-hat-masked")
    # plt.imshow(args["x"], cmap='gray')
    # plt.show()
    # plt.savefig("x")


if __name__ == "__main__":
    main()