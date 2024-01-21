import sys
sys.path.append('../../straightening_models_code/PooledStatisticsMetamers/poolstatmetamer/')

import poolingregions as pyrpool
import metamersolver as pyrsolver
import torch
import torch.nn as nn

class BruceNet(nn.Module):
    'Forward Pass of Bruces Metamer Generator Network'
    def __init__(self, pooling_region_size, 
                 pyramid_params, pyramid_stats=None,
                 dummy_img=None):
        super().__init__()
        
        if dummy_img is None:
            dummy_img = torch.rand((1,1,512,512))
        self.pool = pyrpool.make_uniform_pooling(pooling_region_size)
        self.solver = pyrsolver.make_solver(target_image=dummy_img,
                                          stat_pooling=self.pool, 
                                          pyramid_params=pyramid_params)
        if (pyramid_stats is not None):
            stateval = self.solver.get_statistics_evaluator()
            stateval.set_all_stats(False)
            stateval.set_stat(pyramid_stats,True)
        
    def calcstats(self, current_frame, get_labels=False):
        stats = self.solver.stat_eval(current_frame, create_labels=get_labels)
        if(get_labels):
            labels = self.solver.stat_eval.get_all_labels()
            return(labels)
        else:
            #print('len',len(stats))
            #print([s.shape for s in stats])
            #will need to change this for color, dim is removed as is
            stats = torch.stack([s.squeeze() for s in stats])
            #print('stsh',stats.shape)#.squeeze(2) #brucenet passes a list of size numstats
            if(stats.dim()==1):
                return(stats)
            else:
                stats = torch.transpose(stats,1,0) #transpose batch and list dims
                #print('stats shape:',stats.shape)
                #print(len(stats),stats[1])
                return(stats)
        
    def forward(self, x):
        stats = self.calcstats(x)
        return(stats)