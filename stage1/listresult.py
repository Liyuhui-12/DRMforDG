def printlist(x):
    for i in x:
        print(i)

import os
import argparse


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='result')
    # parser.add_argument('--in_dir', type=str, default='001_net_1_20_erm000')
    # parser.add_argument('--out_dir', type=str, default='/home/xxx/code/ood/dn/weights')
    parser.add_argument('--in_dir', type=str, default='/data2/xxx/code/ood/stage1/ds_xy')
    parser.add_argument('--out_dir', type=str, default='/data2/xxx/code/ood/stage2/weights')
    parser.add_argument('--not_copy', action='store_true')
    args = parser.parse_args()

    path='000_Terr_t0'

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    res_dict={}
    for i in os.listdir(args.in_dir):
        bestpath=os.path.join(args.in_dir,i,'best.txt')
        try:
            with open(bestpath, 'r') as f:
                fl=f.readlines()
            dataset=fl[0][:-1]
            target=fl[1][:-1]
            best=fl[2][:-1]
            dt=dataset+'_target_'+target
            if dt not in res_dict.keys():
                res_dict[dt]=[]
            res_dict[dt].append((os.path.join(args.in_dir,i),best))
        except:
            print('empty')


    #sort
    for i in res_dict:
        res_dict[i].sort(key=lambda x:x[1])

    print('after sort')

    for i in res_dict:
        print(i)
        printlist(res_dict[i])


    if not args.not_copy:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        for i in res_dict:
            weightpath=res_dict[i][-1][0]
            print('copy',res_dict[i][-1])
            #print(weightpath)
            os.system('cp {}/model_{}.ckpt {}'.format(weightpath,i,args.out_dir))







