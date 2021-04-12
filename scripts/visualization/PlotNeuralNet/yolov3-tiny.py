import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

init_width = 416
init_chn = 16

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    #input
    to_input( './examples/fcn8s/cats.jpg', to='(-6,0,0)', width=8.2, height=8.2 , name="inputs"  ),
    
    # Extra Layer - data transfer
    to_Extra("Extra", offset="(-4,0,0)", to="(0,0,0)", width=20, height=4, depth=4, caption="Extra" ),
    
    # Conv
    to_Conv("Conv", init_width, init_chn, offset="(4,0,0)", to="(Extra-east)", height=int(init_width/10), depth=int(init_width/10), width=4, caption="Fused Conv Layer 0"),
    to_Pool("pool2", offset="(0,0,0)", to="(Conv-east)", width=1, height=int(init_width/(10+3)), depth=int(init_width/(10+3)), opacity=0.5, caption=""),
    to_connection("Extra", "Conv"),
    
    # Conv_1
    to_Conv("Conv_1", int(init_width/2), int(init_chn*2), offset="(4,0,0)", to="(pool2-east)", height=int(init_width/12), depth=int(init_width/12), width=4, caption="Fused Conv Layer 1" ),
    to_Pool("pool2_1", offset="(0,0,0)", to="(Conv_1-east)", width=1, height=int(init_width/(12+3)), depth=int(init_width/(12+3)), opacity=0.5, caption=""),
    
    to_connection("pool2", "Conv_1"),
    
    # Conv_2
    to_Conv("Conv_2", int(init_width/4), int(init_chn*4), offset="(4,0,0)", to="(pool2_1-east)", height=int(init_width/15), depth=int(init_width/15), width=4, caption="Fused Conv Layer 2" ),
    to_Pool("pool2_2", offset="(0,0,0)", to="(Conv_2-east)", width=1, height=int(init_width/(15+3)), depth=int(init_width/(15+3)), opacity=0.5, caption=" "),
    
    to_connection("pool2_1", "Conv_2"),
    
    # Conv_3
    to_Conv("Conv_3", int(init_width/8), int(init_chn*8), offset="(4,0,0)", to="(pool2_2-east)", height=int(init_width/18), depth=int(init_width/18), width=4, caption="Fused Conv Layer 3" ),
    to_Pool("pool2_3", offset="(0,0,0)", to="(Conv_3-east)", width=1, height=int(init_width/(18+3)), depth=int(init_width/(18+3)), opacity=0.5, caption=" "),
    
    to_connection("pool2_2", "Conv_3"),
    
    # Conv_4
    to_Conv("Conv_4", int(init_width/16), int(init_chn*16), offset="(4,0,0)", to="(pool2_3-east)", height=int(init_width/21), depth=int(init_width/21), width=4, caption="Fused Conv Layer 4" ),
    to_Pool("pool2_4", offset="(0,0,0)", to="(Conv_4-east)", width=1, height=int(init_width/(21+3)), depth=int(init_width/(21+3)), opacity=0.5, caption=" "),
    
    to_connection("pool2_3", "Conv_4"),
    
    # Conv_5
    to_Conv("Conv_5", int(init_width/32), int(init_chn*32), offset="(4,0,0)", to="(pool2_4-east)", height=int(init_width/24), depth=int(init_width/24), width=4, caption="Fused Conv Layer 5" ),
    to_Pool("pool2_5", offset="(0,0,0)", to="(Conv_5-east)", width=1, height=int(init_width/(24+3)), depth=int(init_width/(24+3)), opacity=0.5, caption=" "),
    
    to_connection("pool2_4", "Conv_5"),
    
    # Conv_6 - deviation from previous layers
    to_Conv("Conv_6", int(init_width/32), int(init_chn*64), offset="(4,0,0)", to="(pool2_5-east)", height=int(init_width/24), depth=int(init_width/24), width=4, caption="Conv Layer 6" ),
    
    to_connection("pool2_5", "Conv_6"),
    
    # Conv_7
    to_Conv("Conv_7", int(init_width/32), int(init_chn*16), offset="(4,0,0)", to="(Conv_6-east)", height=int(init_width/24), depth=int(init_width/24), width=4, caption="Conv Layer 7" ),
    
    to_connection("Conv_6", "Conv_7"),
    
    # Conv_8
    to_Conv("Conv_8", int(init_width/32), int(init_chn*32), offset="(4,0,0)", to="(Conv_7-east)", height=int(init_width/24), depth=int(init_width/24), width=4, caption="Conv Layer 8" ),
    
    to_connection("Conv_7", "Conv_8"),
    
    # Conv_9
    to_Conv("Conv_9", int(init_width/32), int(init_chn*16)-1, offset="(4,0,0)", to="(Conv_8-east)", height=int(init_width/24), depth=int(init_width/24), width=4, caption="Conv Layer 9" ),
    
    to_connection("Conv_8", "Conv_9"),
    
    # RegionYolo
    to_RegionYolo("RegionYolo", offset="(4,0,0)", to="(Conv_9-east)", height=8, depth=8, width=8, caption="RegionYolo" ),
    
    to_connection("Conv_9", "RegionYolo"),
    # END OF THIS PATH
    
    # Conv_10
    to_Conv("Conv_10", int(init_width/32), int(init_chn*16), offset="(8,-5,0)", to="(Conv_9-east)", height=int(init_width/24), depth=int(init_width/24), width=4, caption="Conv Layer 10" ),
    
    to_connection("Conv_9", "Conv_10"),
    
    # Resize
    to_Resample("ResizeNearestNeighbor", offset="(13,-5,0)", to="(Conv_9-east)", height=15, depth=15, width=6, caption="Resize Nearest Neighbor" ),
    
    to_connection("Conv_10", "ResizeNearestNeighbor"),
    
    # Conv_11
    to_Conv("Conv_11", int(init_width/32), int(init_chn*16), offset="(4,0,0)", to="(ResizeNearestNeighbor-east)", height=int(init_width/24), depth=int(init_width/24), width=4, caption="Conv Layer 11" ),
    
    to_connection("Conv_4", "Conv_11"),
    to_connection("ResizeNearestNeighbor", "Conv_11"),
    
    # Conv_12
    to_Conv("Conv_12", int(init_width/32), int(init_chn*16)-1, offset="(4,0,0)", to="(Conv_11-east)", height=int(init_width/24), depth=int(init_width/24), width=4, caption="Conv Layer 12" ),
    
    to_connection("Conv_11", "Conv_12"),
    
    # RegionYolo2
    to_RegionYolo("RegionYolo2", offset="(4,0,0)", to="(Conv_12-east)", height=8, depth=8, width=8, caption="RegionYolo2" ),
    
    to_connection("Conv_12", "RegionYolo2"),
    # END OF THIS PATH
    
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
"""
% input picture
\node[canvas is zy plane at x=0] (inputs) at (-4,0,0) {\includegraphics[width=8cm,height=8cm]{./examples/fcn8s/cats.jpg}};

% Conv

\pic[shift={(0,0,0)}] at (0,0,0)
    {Box={
        name=Conv,
        caption=Fused Conv Layer 1,
        xlabel={{16, }},
        ylabel=416,
        zlabel=416,
        fill=\ConvColor,
        height=41,
        width=4,
        depth=41
        }
    };



\pic[shift={ (0,0,0) }] at (Conv-east)
    {Box={
        name=pool2,
        caption=,
        fill=\PoolColor,
        opacity=0.5,
        height=32,
        width=1,
        depth=32
        }
    };

% Conv_1

\pic[shift={(3,0,0)}] at (pool2-east)
    {Box={
        name=Conv_1,
        caption=Fused Conv Layer 2,
        xlabel={{32, }},
        ylabel=208,
        zlabel=208,
        fill=\ConvColor,
        height=20,
        width=4,
        depth=20
        }
    };
    


\pic[shift={ (0,0,0) }] at (Conv_1-east)
    {Box={
        name=pool2_1,
        caption= ,
        fill=\PoolColor,
        opacity=0.5,
        height=16,
        width=1,
        depth=16
        }
    };

\draw [connection] (pool2-east)    -- node {\midarrow} (Conv_1-west);

% Conv_2

\pic[shift={(6,0,0)}] at (pool2-east)
    {Box={
        name=Conv_2,
        caption=Fused Conv Layer 3,
        xlabel={{64, }},
        ylabel=104,
        zlabel=104,
        fill=\ConvColor,
        height=10,
        width=4,
        depth=10
        }
    };
    


\pic[shift={ (0,0,0) }] at (Conv_2-east)
    {Box={
        name=pool2_2,
        caption= ,
        fill=\PoolColor,
        opacity=0.5,
        height=8,
        width=1,
        depth=8
        }
    };

\draw [connection] (pool2_1-east)    -- node {\midarrow} (Conv_2-west);

% Conv_3

\pic[shift={(3,0,0)}] at (pool2_2-east)
    {Box={
        name=Conv_3,
        caption=Fused Conv Layer 4,
        xlabel={{128, }},
        ylabel=52,
        zlabel=52,
        fill=\ConvColor,
        height=5,
        width=4,
        depth=5
        }
    };
    


\pic[shift={ (0,0,0) }] at (Conv_3-east)
    {Box={
        name=pool2_3,
        caption= ,
        fill=\PoolColor,
        opacity=0.5,
        height=4,
        width=1,
        depth=4
        }
    };

\draw [connection] (pool2_2-east)    -- node {\midarrow} (Conv_3-west);

"""