#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

##################################################################################################
from re import L
import wandb
import numpy as np
from datetime import datetime
import time
from lpipsPyTorch import lpips
from lpips import lpips
from utils.sh_utils import SH2RGB
##################################################################################################

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import lpips
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import dbscan
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Gaussian 모델을 사용하여 렌더링 작업을 위한 훈련을 수행
# 매개변수:
# dataset: 이미지와 카메라 매개변수를 포함한 훈련 데이터.
# opt: 최적화 매개변수.
# pipe: 파이프라인 매개변수.
# testing_iterations: 테스트를 수행할 반복 횟수.
# saving_iterations: 모델을 저장할 반복 횟수.
# checkpoint_iterations: 체크포인트를 저장할 반복 횟수.
# checkpoint: 훈련을 재개할 체크포인트 파일 경로.
# debug_from: 디버깅을 시작할 반복 횟수.
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, dbscan_iteration):
    # [함수 정의 및 초기 설정] # 
    first_iter = 0 # 첫 번째 반복 초기화
    tb_writer = prepare_output_and_logger(dataset) # TensorBoard 작성기를 준비하여 로그
    gaussians = GaussianModel(dataset.sh_degree) # Gaussian 모델을 생성
    scene = Scene(dataset, gaussians) # 데이터셋과 Gaussian 모델을 사용하여 장면을 초기화
    gaussians.training_setup(opt) # Gaussian 모델의 훈련 매개변수를 설정

    # 체크포인트가 있는 경우 체크포인트를 로드하고 모델 매개변수와 반복을 복원
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 데이터셋 설정에 따라 배경 색상을 설정
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 반복 시간을 측정하기 위해 CUDA 이벤트를 생성
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ##################################################################################################
    total_start_time = time.time()  # 전체 훈련 시작 시간
    ##################################################################################################

    # [주요 훈련 루프] #
    viewpoint_stack = None # 뷰포인트 스택을 초기화
    ema_loss_for_log = 0.0 # 로그를 위한 손실의 지수 이동 평균을 초기화
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") # 진행 표시줄을 초기화
    first_iter += 1 # 첫 번째 반복을 증가

    ##################################################################################################
    memory_usage = []
    ##################################################################################################

    for iteration in range(first_iter, opt.iterations + 1):   # 훈련 반복을 실행 
        # 네트워크 GUI 연결이 없으면 연결을 시도   
        if network_gui.conn == None:
            network_gui.try_connect()
        # 네트워크 GUI 연결을 통해 데이터를 수신 및 송신하고, 필요한 경우 훈련을 수행
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record() # 현재 반복의 시작 시간을 기록

        gaussians.update_learning_rate(iteration) # 현재 반복에 따라 학습률을 업데이트

        # Every 1000 its we increase the levels of SH up to a maximum degree (1000번마다 SH(구면 고조파)의 레벨을 최대 차수까지 증가)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera (훈련용 카메라 중에서 랜덤하게 하나를 선택)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # =============================================
        # 이 부분에서 DBSCAN 수행 및 변수 초기화
        # iteration 지정해줌
        if iteration in dbscan_iteration:
            print('\nSET PARAMETER....')
            pcl, opacities = dbscan.set_parameter(gaussians)

            for _ in tqdm(range(1), desc='DBSCAN....'):
              labels = dbscan.dbscan(pcl)
            points, xyz, f_dc = dbscan.cluster_mean(pcl,labels)
            scales,rots = dbscan.scale_rotaion(points,labels)
            opacities = dbscan.opacities_cluster(opacities,labels)

            gaussians.set_dbscan_parameter(xyz,f_dc,scales,rots,opacities)
            print('\n========FINISH========\n')
        # =============================================


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # bg = torch.rand((3), device="cuda") if opt.random_background else background # 랜덤 배경을 사용할지 설정된 배경을 사용할지 결정

        # 선택된 카메라로 이미지를 렌더링하고 결과를 저장
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss: (1-lambda)*L1 + lambda*L_{D-SSIM} (손실 함수(L1 손실과 SSIM 손실)를 계산하고 역전파를 수행)
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record() # 현재 반복의 종료 시간을 기록

        with torch.no_grad(): # 모델의 평가가 아닌 부수적인 작업을 수행하는 데 사용

            # Progress bar (손실의 지수 이동 평균을 업데이트하고, 진행 표시줄에 손실 값을 표시)
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save (TensorBoard와 W&B에 훈련 진행 상황을 보고)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            ##################################################################################################
            ##  GPU 메모리 사용량 로깅
            # current_memory_usage: 현재 GPU에서 할당된 메모리 용량. 이는 현재 시점에서 GPU에 할당된 실제 메모리 양
            # max_memory_usage: 현재까지 측정된 GPU의 최대 메모리 사용량. 즉, 프로그램 실행 중에 GPU가 사용한 최대 메모리 양입니다.
            current_memory_usage = torch.cuda.memory_allocated() / (1024**2)  # MB 단위로 변환
            max_memory_usage = torch.cuda.max_memory_allocated() / (1024**2)  # MB 단위로 변환
            wandb.log({"GPU Memory Usage (MB)": current_memory_usage,"iteration": iteration})
            wandb.log({"MAX GPU Memory Usage (MB)": max_memory_usage,"iteration": iteration})

            memory_usage.append([current_memory_usage,max_memory_usage])
            ##################################################################################################

            # 저장할 반복 횟수에 도달하면 모델을 저장
            if (iteration in saving_iterations): 
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification (모델의 밀도를 높이고 불필요한 부분을 제거하는 작업)
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # visibility_filter에 해당하는 Gaussian의 이미지 공간에서 최대 반지름을 추적
                # visibility_filter를 만족하는 Gaussian의 최대 반지름을 업데이트
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 밀도화 통계 정보 추가. 이는 Gaussian splatting 과정에서 어떤 점들이 어떻게 밀도화되는지 통계를 기록하는 단계
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 특정 조건일 때, densify한 후 pruning까지
                if iteration > opt.densify_from_iter1 and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # 투명도 초기화
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step (최적화 단계와 경사 초기화를 수행)
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 체크포인트 저장 반복 횟수에 도달하면 현재 모델 매개변수와 반복 수를 저장
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            
    ##################################################################################################
    # 전체 훈련 시간 계산 및 로깅 (수정ver)
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    wandb.log({"total_training_time": total_training_time, "iteration": iteration})

    # 전체 메모리 사용량 계산 및 로깅
    total_memory_usage = sum(x[0] for x in memory_usage)
    #avg_max_memory_usage = sum(x[1] for x in memory_usage) / len(memory_usage)
    #wandb.log({"average_current_memory_usage": avg_current_memory_usage, "average_max_memory_usage": avg_max_memory_usage})
    wandb.log({"total_memory_usage": total_memory_usage, "iteration": iteration})
    ##################################################################################################
 


def prepare_output_and_logger(args):    
    # 모델 경로가 지정되지 않았으면, OAR_JOB_ID 환경 변수에서 가져오거나 uuid를 생성하여 설정
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder (출력 폴더 설정)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # # 인수(args)를 "cfg_args" 파일로 저장
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer (TensorBoard 작성자를 생성)
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


##################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='vgg').to(device)

# Update the training_report function to log to wandb
# render_table = wandb.Table(columns=["iteration", "viewpoint", "render_image", "gt_image"])
##################################################################################################
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    
    ##################################################################################################
# # TensorBoard 작성자가 있다면, L1 손실, 전체 손실, 반복 시간 로그를 작성
    # if tb_writer:
    #     tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
    #     tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
    #     tb_writer.add_scalar('iter_time', elapsed, iteration)

    # wandb에 로그 작성 (L1 손실, 전체 손실, 반복 시간)
    wandb.log({"l1_loss": Ll1.item(), "iteration": iteration})
    wandb.log({"total_loss": loss.item(), "iteration": iteration})
    wandb.log({"iter_time": elapsed, "iteration": iteration})
    ##################################################################################################

    # Report test and samples of training set (테스트 및 학습 셋 샘플 평가)
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        ##################################################################################################
        # 렌더링 속도(FPS) 측정을 위한 시작 시간 기록
        render_start_time = time.time()
        ##################################################################################################


        # 검증 구성 설정 (테스트 카메라 및 학습 카메라)
        # Tanks and Temples에서는 테스트 카메라 없기 때문에 train data 중에서 임의로 5개 뽑음
        test_cameras = scene.getTestCameras()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        ##################################################################################################
        render_table = wandb.Table(columns=["iteration", "viewpoint", "render_image", "gt_image"])
        ##################################################################################################
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ##################################################################################################
                ssims = []
                lpipss = []
                render_images = []
                gt_images = []
                ##################################################################################################
                for idx, viewpoint in enumerate(config['cameras']):
                    # 이미지를 렌더링하고 GT 이미지와 비교
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    ##################################################################################################
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    # Convert image to numpy array with correct shape and type
                    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    gt_image_np = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    
                    if idx < 5:
                      render_table.add_data(iteration, f"{config['name']}_view_{viewpoint.image_name}", wandb.Image(image_np), wandb.Image(gt_image_np))
          
                      # render_images.append(wandb.Image(image_np, caption=f"Render {idx}"))
                      # gt_images.append(wandb.Image(gt_image_np, caption=f"Ground Truth {idx}"))
                    ##################################################################################################
                    ##################################################################################################
                    # L1 손실 및 PSNR 계산
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    # SSIM, LPIPS 계산
                    ssims.append(ssim(image, gt_image))
                    #lpipss.append(lpips(image, gt_image, net_type='vgg'))

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    lpips_model = lpips.LPIPS(net='vgg').to(device)

                    lpips_value = lpips_model(image, gt_image)
                    lpipss.append(lpips_value.item())  # .item()을 사용하여 텐서를 숫자로 변환
                    ##################################################################################################
                
                

                ##################################################################################################               
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()

                # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                print("\n[ITER {}] Evaluating {}: L1 {}".format(iteration, config['name'], l1_test))
                
                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean(), ".5"))
                print("  LPIPS : {:>12.7f}".format(lpipss_test.mean(), ".5"))
                print("")
                ##################################################################################################
                # # 불투명도 히스토그램과 전체 포인트 로그
                # if tb_writer:
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                
                #wandb.log({"render_table": render_table})

                wandb.log({config['name'] + '/loss_viewpoint - l1_loss': l1_test,"iteration": iteration})
                wandb.log({config['name'] + '/loss_viewpoint - psnr': psnr_test,"iteration": iteration})
                wandb.log({config['name'] + '/loss_viewpoint - ssim': ssims_test,"iteration": iteration})
                wandb.log({config['name'] + '/loss_viewpoint - lpip': lpipss_test,"iteration": iteration})

                # wandb.log({config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                #            config['name'] + '/loss_viewpoint - psnr': psnr_test, 
                #            config['name'] + '/loss_viewpoint - ssim': ssims_test,
                #            config['name'] + '/loss_viewpoint - lpip': lpipss_test
                          #  config['name'] + '_render_images': render_images,
                          #  config['name'] + '_gt_images': gt_images
                          # })
                ##################################################################################################



        # 테이블 저장
        wandb.log({f"scene/render_table_{iteration}": render_table})
        ##################################################################################################
        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        

        wandb.log({"scene/opacity_histogram": wandb.Histogram(scene.gaussians.get_opacity.cpu()), "iteration": iteration})
        wandb.log({"total_points": scene.gaussians.get_xyz.shape[0],"iteration": iteration})

        # 렌더링 속도(FPS) 측정을 위한 종료 시간 기록
        render_end_time = time.time()
        # 렌더링 소요 시간 계산
        render_time = (render_end_time - render_start_time) / len(config['cameras'])
        wandb.log({"render_time": render_time, "iteration": iteration})

        # FPS 계산
        # # FPS 렌더링 속도는 초당 처리된 프레임 수
        # fps = len(config['cameras']) / render_time  # 한 번에 처리된 카메라 수를 렌더링 시간으로 나누어 초당 FPS 계산
        # # FPS 로그 기록
        # wandb.log({"render_fps": fps, "iteration": iteration})


        # 테이블 저장
        # wandb.log({"scene/render_table": render_table})
        ##################################################################################################
        # # scene.gaussians에서 점들의 위치를 추출하여 points 배열로 구성
        # points = scene.gaussians.get_xyz.cpu().numpy() # GPU에서 CPU로 데이터 이동 후 numpy 배열로 변환
        # pointSize = scene.gaussians.get_scaling.cpu().numpy()
        # opacity = scene.gaussians.get_opacity.cpu().numpy()

        # # 점들의 위치와 관련된 데이터를 포함하는 Object3D 객체 생성
        # point_cloud = wandb.Object3D(
        # {
        #   "points": {points[0],points[1],points[2]},
        #   # 추가적인 설정(선택 사항):
        #   #"pointSize": {pointSize},  # 점의 크기 설정
        #   #"colors": scene.gaussians.get_opacity.cpu()  # 각 점의 색상 설정
        #   #"opacity": {opacity}  # 점의 투명도 설정
        # })

        # # Point cloud를 WandB에 로그
        # wandb.log({"scene/point_cloud": point_cloud})
        ##################################################################################################
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    ##################################################################################################
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000, 11_000, 12_000, 13_000, 14_000, 15_000, 16_000, 17_000, 18_000, 19_000, 20_000, 21_000, 22_000, 23_000, 24_000, 25_000, 26_000, 27_000, 28_000, 29_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    ##################################################################################################
    parser.add_argument("--dbscan_iteration", type=int, default=[3000, 10000, 20000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    ##################################################################################################
    # WandB API 키와 엔터티를 환경 변수에서 가져오기
    wandb_api = os.getenv('WANDB_API_KEY')
    wandb_entity = os.getenv('WANDB_ENTITY')

    # set wandb logger
    now = datetime.now().strftime('%H%M%S_%y%m%d')
    
    wandb.init(
        project='deepdaiv',
        entity=wandb_entity
    )
    wandb.run.name = f'dbscan_{now}'
    ##################################################################################################
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG) (# 시스템 상태 초기화 (RNG))
    safe_state(args.quiet)

    # Start GUI server, configure and run training (GUI 서버 시작, 설정 및 훈련 실행)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.dbscan_iteration)

    # All done (모든 작업 완료)
    print("\nTraining complete.")
