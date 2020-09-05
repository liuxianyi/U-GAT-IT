import time, itertools,os,cv2
import numpy as np
from dataset import DatasetFolder
import hapi.vision.transforms as transforms
import paddle
import paddle.fluid as fluid 
from paddle.fluid import dygraph as nn 
from paddle.fluid.dygraph import to_variable
from networks import *
from utils import *
from nn import *
# from glob import glob

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light #False

        if self.light : 
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT' #FASLE

        self.result_dir = args.result_dir # results
        self.dataset = args.dataset # YOUR_DATASET_NAME

        self.iteration = args.iteration # 1000000
        self.decay_flag = args.decay_flag # True

        self.batch_size = args.batch_size # 1
        self.print_freq = args.print_freq # 1000
        self.save_freq = args.save_freq # 100000

        self.lr = args.lr # 0.0001
        self.weight_decay = args.weight_decay # 0.0001
        self.ch = args.ch # 64

        """ Weight """
        self.adv_weight = args.adv_weight # 1
        self.cycle_weight = args.cycle_weight # 10
        self.identity_weight = args.identity_weight # 10
        self.cam_weight = args.cam_weight # 1000

        """ Generator """
        self.n_res = args.n_res # 4

        """ Discriminator """
        self.n_dis = args.n_dis # 6

        self.img_size = args.img_size # 256
        self.img_ch = args.img_ch # 3

        self.device = args.device # cuda
        self.benchmark_flag = args.benchmark_flag # False
        self.resume = args.resume # False

        # if torch.backends.cudnn.enabled and self.benchmark_flag:
        #     print('set benchmark !')
        #     torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomResizedCrop((self.img_size, self.img_size))
            
        ])
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            
        ])
        def generator_creator(items):
            def __reader__():
                for item in items:

                    yield np.array(item[0]).astype('float32').transpose(2,0,1)
            return __reader__
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.trainA = DatasetFolder(os.path.join('data', self.dataset, 'trainA'), transform=train_transform)

        self.trainB = DatasetFolder(os.path.join('data', self.dataset, 'trainB'), transform=train_transform)
        self.testA = DatasetFolder(os.path.join('data', self.dataset, 'testA'), transform=test_transform)
        self.testB = DatasetFolder(os.path.join('data', self.dataset, 'testB'), transform=test_transform)

        self.trainA_loader = paddle.fluid.io.shuffle(paddle.fluid.io.batch(generator_creator(self.trainA),
                                                     self.batch_size),20)
        self.trainB_loader = paddle.fluid.io.shuffle(paddle.fluid.io.batch(generator_creator(self.trainB),
                                                     self.batch_size),20)
        self.testA_loader = paddle.fluid.io.batch(generator_creator(self.testA),
                                                     batch_size=1)
        self.testB_loader = paddle.fluid.io.batch(generator_creator(self.testB),
                                                     batch_size=1)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        """ Define Loss """
        self.L1_loss = L1Loss()
        self.MSE_loss = MSELoss()
        self.BCE_loss = BCEWithLogitsLoss()

        """ Trainer """
        def fileter_func(paramBase):
            if paramBase.name == 'rho10':
                return True
            if paramBase.name == 'rho11':
                return True
            return False


        self.G_optim = fluid.optimizer.Adam(learning_rate=self.lr,
        beta1=0.5,
        beta2=0.999,
        parameter_list=self.genA2B.parameters()+self.genB2A.parameters(), 
        regularization=paddle.fluid.regularizer.L2Decay(
            regularization_coeff=self.weight_decay
            ),
            
            )
            #grad_clip=fluid.clip.GradientClipByValue(min=0, max=1, need_clip=fileter_func)
        self.D_optim = fluid.optimizer.Adam(learning_rate=self.lr,
        beta1=0.5,
        beta2=0.999,
        parameter_list=self.disGA.parameters()+self.disGB.parameters()+self.disLA.parameters()+self.disLB.parameters(), 
        regularization=paddle.fluid.regularizer.L2Decay(
            regularization_coeff=self.weight_decay,
            
            ))
        #,grad_clip=fluid.clip.GradientClipByValue(min=0, max=1, need_clip=fileter_func)
        # self.G_optim_A2B = fluid.optimizer.Adam(learning_rate=self.lr,
        # beta1=0.5,
        # beta2=0.999,
        # parameter_list=self.genA2B.parameters(), 
        # regularization=paddle.fluid.regularizer.L2Decay(
        #     regularization_coeff=self.weight_decay
        #     ),
        # grad_clip=fluid.clip.GradientClipByValue(1, 0)
        #     )
        # self.G_optim_B2A = fluid.optimizer.Adam(learning_rate=self.lr,
        # beta1=0.5,
        # beta2=0.999,
        # parameter_list=self.genB2A.parameters(), 
        # regularization=paddle.fluid.regularizer.L2Decay(
        #     regularization_coeff=self.weight_decay
        #     ),
        # grad_clip=fluid.clip.GradientClipByValue(1, 0)
        #     )

    
        # self.D_optim_GA = fluid.optimizer.Adam(learning_rate=self.lr,
        # beta1=0.5,
        # beta2=0.999,
        # parameter_list=self.disGA.parameters(), 
        # regularization=paddle.fluid.regularizer.L2Decay(
        #     regularization_coeff=self.weight_decay
        #     ),
        # grad_clip=fluid.clip.GradientClipByValue(1, 0))
        # self.D_optim_GB = fluid.optimizer.Adam(learning_rate=self.lr,
        # beta1=0.5,
        # beta2=0.999,
        # parameter_list=self.disGB.parameters(), 
        # regularization=paddle.fluid.regularizer.L2Decay(
        #     regularization_coeff=self.weight_decay
        #     ),
        # grad_clip=fluid.clip.GradientClipByValue(1, 0))

        # self.D_optim_LA = fluid.optimizer.Adam(learning_rate=self.lr,
        # beta1=0.5,
        # beta2=0.999,
        # parameter_list=self.disLA.parameters(), 
        # regularization=paddle.fluid.regularizer.L2Decay(
        #     regularization_coeff=self.weight_decay
        #     ),
        # grad_clip=fluid.clip.GradientClipByValue(1, 0))
        # self.D_optim_LB = fluid.optimizer.Adam(learning_rate=self.lr,
        # beta1=0.5,
        # beta2=0.999,
        # parameter_list=self.disLB.parameters(), 
        # regularization=paddle.fluid.regularizer.L2Decay(
        #     regularization_coeff=self.weight_decay
        #     ),
        # grad_clip=fluid.clip.GradientClipByValue(1, 0))
        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        #self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        # print(self.trainA[0][0].shape)
        start_iter = 1
        if self.resume: # 加载训练参数
            self.load(self.result_dir, 10000)
            print(" [*] Load SUCCESS")
        #     model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        #     if not len(model_list) == 0:
        #         model_list.sort()
        #         start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
        #         self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
        #         print(" [*] Load SUCCESS")
        #         if self.decay_flag and start_iter > (self.iteration // 2):
        #             self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
        #             self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
        #https://github.com/PaddlePaddle/PaddleGAN/blob/master/ppgan/solver/lr_scheduler.py#L9
        # training loop
        print('training start !')
       

        
        for step in range(start_iter, self.iteration + 1):
            start_time = time.time()
        #     if self.decay_flag and step > (self.iteration // 2):
                #print(self.G_optim.state_dict())
                # self.G_optim.state_dict()[0]['lr'] -= (self.lr / (self.iteration // 2))
                # self.D_optim.state_dict()[0]['lr'] -= (self.lr / (self.iteration // 2))
                # lr =  self.G_optim_A2B.current_step_lr()
                # np.allclose(lr, lr-(self.lr/(self.iteration+1)), rtol=1e-06, atol=0.0) # True
                # lr =  self.G_optim_B2A.current_step_lr()
                # np.allclose(lr, lr-(self.lr/(self.iteration+1)), rtol=1e-06, atol=0.0) # True
                # lr =  self.D_optim_GA.current_step_lr()
                # np.allclose(lr, lr-(self.lr/(self.iteration+1)), rtol=1e-06, atol=0.0) # True
                # lr =  self.D_optim_GB.current_step_lr()
                # np.allclose(lr, lr-(self.lr/(self.iteration+1)), rtol=1e-06, atol=0.0) # True
                # lr =  self.D_optim_LA.current_step_lr()
                # np.allclose(lr, lr-(self.lr/(self.iteration+1)), rtol=1e-06, atol=0.0) # True
                # lr =  self.D_optim_LB.current_step_lr()
                # np.allclose(lr, lr-(self.lr/(self.iteration+1)), rtol=1e-06, atol=0.0) # True

            try:
                real_A= next(trainA_iter)
            except:
                trainA_iter = iter(self.trainA_loader())
                real_A= next(trainA_iter)

            try:
                real_B= next(trainB_iter)
            except:
                trainB_iter = iter(self.trainB_loader())
                real_B= next(trainB_iter)
            
            real_A, real_B = to_variable(np.array(real_A).astype('float32')), to_variable(np.array(real_B).astype('float32'))
            
                
            # Update D
            #print(real_A.shape)
            fake_A2B, _, _ = self.genA2B(real_A)
            # print(real_A.shape)
            # print(fake_A2B.shape)
            #print(fake_A2B.shape)
            fake_B2A, _, _ = self.genB2A(real_B)
            #print(fake_B2A.shape)
            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)
            #fake_B2A.stop_gradient =True
            #fake_A2B.stop_gradient=True
            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
            # print(real_GA_logit.shape)
            # print(fake_GB_logit.shape)
            #test = self.MSE_loss(real_GA_logit, fluid.layers.ones_like(real_GA_logit))
            #test1 = self.MSE_loss(fake_LA_logit, fluid.layers.zeros_like(fake_LA_logit))
            D_ad_loss_GA = self.MSE_loss(real_GA_logit, fluid.layers.ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit, fluid.layers.zeros_like(fake_GA_logit))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, fluid.layers.ones_like(real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit, fluid.layers.zeros_like(fake_GA_cam_logit))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, fluid.layers.ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit, fluid.layers.zeros_like(fake_LA_logit))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, fluid.layers.ones_like(real_LA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, fluid.layers.zeros_like(fake_LA_cam_logit))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, fluid.layers.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit, fluid.layers.zeros_like(fake_GB_logit))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, fluid.layers.ones_like(real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit, fluid.layers.zeros_like(fake_GB_cam_logit))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, fluid.layers.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit, fluid.layers.zeros_like(fake_LB_logit))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, fluid.layers.ones_like(real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, fluid.layers.zeros_like(fake_LB_cam_logit))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss =D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.minimize(Discriminator_loss)
            self.D_optim.clear_gradients()
            # 
            
            # Discriminator_lossGA = fluid.layers.mean(self.adv_weight*(D_ad_loss_GA + D_ad_cam_loss_GA))
            # Discriminator_lossGA.backward()
            # self.D_optim_GA.minimize(Discriminator_lossGA)
            # self.disGA.clear_gradients()
            
            # Discriminator_lossGB = fluid.layers.mean(self.adv_weight*(D_ad_loss_GB + D_ad_cam_loss_GB))
            # Discriminator_lossGB.backward()
            # self.D_optim_GB.minimize(Discriminator_lossGB)
            # self.disGB.clear_gradients()

            
            # Discriminator_lossLA = fluid.layers.mean(self.adv_weight*(D_ad_loss_LA + D_ad_cam_loss_LA))
            # Discriminator_lossLA.backward()
            # self.D_optim_LA.minimize(Discriminator_lossLA)
            # self.disLA.clear_gradients()

            
            # Discriminator_lossLB = fluid.layers.mean(self.adv_weight*(D_ad_loss_LB + D_ad_cam_loss_LB))
            # Discriminator_lossLB.backward()
            # self.D_optim_LB.minimize(Discriminator_lossLB)
            # self.disLB.clear_gradients()
           
            
            
          

            # Update G
            #print('Update G')
            #print(real_A.shape)
            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)
            #print(fake_A2B.shape)
            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)
            #real_A1 = to_variable(np.random.random((2, 3, 256, 256)).astype('float32'))
            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, fluid.layers.ones_like(fake_GA_logit))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, fluid.layers.ones_like(fake_GA_cam_logit))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, fluid.layers.ones_like(fake_LA_logit))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, fluid.layers.ones_like(fake_LA_cam_logit))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, fluid.layers.ones_like(fake_GB_logit))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, fluid.layers.ones_like(fake_GB_cam_logit))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, fluid.layers.ones_like(fake_LB_logit))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, fluid.layers.ones_like(fake_LB_cam_logit))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, fluid.layers.ones_like(fake_B2A_cam_logit)) + self.BCE_loss(fake_A2A_cam_logit, fluid.layers.zeros_like(fake_A2A_cam_logit))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, fluid.layers.ones_like(fake_A2B_cam_logit)) + self.BCE_loss(fake_B2B_cam_logit, fluid.layers.zeros_like(fake_B2B_cam_logit))

            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.minimize(Generator_loss)
            self.G_optim.clear_gradients()

            # Generator_lossA2B = fluid.layers.mean(G_loss_A)
            # Generator_lossA2B.backward()
            # self.G_optim_A2B.minimize(Generator_lossA2B)
            # self.genA2B.clear_gradients()

            # Generator_lossB2A = fluid.layers.mean(G_loss_B)
            # Generator_lossB2A.backward()
            # self.G_optim_B2A.minimize(Generator_lossB2A)
            # self.genB2A.clear_gradients()
            
            # clip parameter of AdaILN and ILN, applied after optimizer step
            # self.genA2B.apply(self.Rho_clipper)
            # self.genB2A.apply(self.Rho_clipper)
            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))

            # print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_lossLA+Discriminator_lossLB+Discriminator_lossGA+Discriminator_lossGB, Generator_lossA2B+Generator_lossB2A))
            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for _ in range(train_sample_num):
                    try:
                        real_A = next(trainA_iter)
                    except:
                        trainA_iter = iter(self.trainA_loader())
                        real_A = next(trainA_iter)

                try:
                    real_B= next(trainB_iter)
                except:
                    trainB_iter = iter(self.trainB_loader())
                    real_B= next(trainB_iter)
                real_A, real_B = to_variable(np.array(real_A).astype('float32')), to_variable(np.array(real_B).astype('float32'))
                #print(real_A.shape)
                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
                #print(fake_A2A[0])
                A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                            cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                            cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                            cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                            cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                            cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                            cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A= next(testA_iter)
                    except:
                        testA_iter = iter(self.testA_loader())
                        real_A= next(testA_iter)

                    try:
                        real_B= next(testB_iter)
                    except:
                        testB_iter = iter(self.testB_loader())
                        real_B= next(testB_iter)
                    real_A, real_B = to_variable(np.array(real_A).astype('float32')), to_variable(np.array(real_B).astype('float32'))
                    
                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % self.save_freq == 0 and step!=0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            # if step % 1000 == 0 and step!=0:
            #     params = {}
            #     params['genA2B'] = self.genA2B.state_dict()
            #     params['genB2A'] = self.genB2A.state_dict()
            #     params['disGA'] = self.disGA.state_dict()
            #     params['disGB'] = self.disGB.state_dict()
            #     params['disLA'] = self.disLA.state_dict()
            #     params['disLB'] = self.disLB.state_dict()
            #     nn.save_dygraph(params['genA2B'], os.path.join(self.result_dir, self.dataset + '_genA2B'))
            #     nn.save_dygraph(params['genB2A'], os.path.join(self.result_dir, self.dataset + '_genB2A'))
            #     nn.save_dygraph(params['disGA'], os.path.join(self.result_dir, self.dataset + '_disGA'))
            #     nn.save_dygraph(params['disGB'], os.path.join(self.result_dir, self.dataset + '_disGB'))
            #     nn.save_dygraph(params['disLA'], os.path.join(self.result_dir, self.dataset + '_disLA'))
            #     nn.save_dygraph(params['disLB'], os.path.join(self.result_dir, self.dataset + '_disLB'))


    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        nn.save_dygraph(params['genA2B'], os.path.join(self.result_dir, self.dataset, 'model', '_genA2B_%07d' % step))
        nn.save_dygraph(params['genB2A'], os.path.join(self.result_dir, self.dataset, 'model', '_genB2A_%07d' % step))
        nn.save_dygraph(params['disGA'], os.path.join(self.result_dir, self.dataset, 'model', '_disGA_%07d' % step))
        nn.save_dygraph(params['disGB'], os.path.join(self.result_dir, self.dataset, 'model', '_disGB_%07d' % step))
        nn.save_dygraph(params['disLA'], os.path.join(self.result_dir, self.dataset, 'model', '_disLA_%07d' % step))
        nn.save_dygraph(params['disLB'], os.path.join(self.result_dir, self.dataset, 'model', '_disLB_%07d' % step))

    def load(self, dir, step):
        
        params = {}
        params['genA2B'], _ = fluid.load_dygraph(os.path.join(dir, self.dataset, 'model', '_genA2B_%07d' % step))
        params['genB2A'], _ = fluid.load_dygraph(os.path.join(dir, self.dataset, 'model', '_genB2A_%07d' % step))
        params['disGA'], _ = fluid.load_dygraph(os.path.join(dir, self.dataset, 'model', '_disGA_%07d' % step))
        params['disGB'], _ = fluid.load_dygraph(os.path.join(dir, self.dataset, 'model', '_disGB_%07d' % step))
        params['disLA'], _ = fluid.load_dygraph(os.path.join(dir, self.dataset, 'model', '_disLA_%07d' % step))
        params['disLB'], _ = fluid.load_dygraph(os.path.join(dir, self.dataset, 'model', '_disLB_%07d' % step))
        self.genA2B.set_dict(params['genA2B'])
        self.genB2A.set_dict(params['genB2A'])
        self.disGA.set_dict(params['disGA'])
        self.disGB.set_dict(params['disGB'])
        self.disLA.set_dict(params['disLA'])
        self.disLB.set_dict(params['disLB'])

    def test(self, step):
        # model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        # if not len(model_list) == 0:
        #     model_list.sort()
        #     iter = int(model_list[-1].split('_')[-1].split('.')[0])
        #     self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [*] Load FAILURE")
        #     return
        params = {}
        params['genA2B'], _ = fluid.load_dygraph(os.path.join(self.result_dir, self.dataset,'model', '_genA2B_%07d' % step))
        params['genB2A'], _ = fluid.load_dygraph(os.path.join(self.result_dir, self.dataset, 'model', '_genB2A_%07d' % step))
        self.genA2B.set_dict(params['genA2B'])
        self.genB2A.set_dict(params['genB2A'])
        self.genA2B.eval(), self.genB2A.eval()
        for n, real_A in enumerate(self.testA_loader()):
            real_A = to_variable(np.array(real_A).astype('float32'))
            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            # A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
            #                       cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
            #                       RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
            #                       cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
            #                       RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
            #                       cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
            #                       RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)
            A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)
            print('test/A2B A2B_%d.png saved' % (n + 1))
        # for n, real_B in enumerate(self.testB_loader()):
        #     real_B = to_variable(np.array(real_B).astype('float32'))
        #     fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

        #     fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

        #     fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
        #     B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))
        #     # B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
        #     #                       cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
        #     #                       RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
        #     #                       cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
        #     #                       RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
        #     #                       cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
        #     #                       RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)
            
        #     cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
        #     print('test/A2B A2B_%d.png saved' % (n + 1))