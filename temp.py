import os
import shutil

root_path = "samples/bsp_ae_out/001/"
gt_mesh_path = "/disk2/occupancy_networks-master/data/ABC.build/001/4_watertight_scaled/"
if not os.path.exists("input"):
    os.mkdir("input")
if not os.path.exists("gt_mesh"):
    os.mkdir("gt_mesh")
for mesh_name in ['00011821',
                      '00017102',
                      '00019656',
                      '00014474',
                      '00012251',
                      '00013988',
                      '00017320',
                      '00012058',
                      '00013729',
                      '00012913',
                      '00010686',
                      '00015363',
                      '00019541',
                      '00016461',
                      '00018146',
                      '00011095']:
    shutil.copy(root_path+mesh_name+"_pc.txt", "input/"+mesh_name+".txt")
    # shutil.copy( "input/" + mesh_name + ".txt", root_path + mesh_name + "_pc.txt")
    shutil.copy(root_path + mesh_name + ".off", "gt_mesh/" + mesh_name + ".off")