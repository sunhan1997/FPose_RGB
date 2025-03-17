记录，

先是调用https://github.com/spacewalk01/depth-anything-tensorrt里面的
python export_v2.py --encoder vitb --input-size 518
生成/home/sunh/6D_ws/Fpose_rgb/feature2/depth_anything_v2_vitl.onnx

然后使用https://blog.csdn.net/qq_39045712/article/details/142857630?ops_request_misc=&request_id=&biz_id=102&utm_term=depthanything%20%20v2%20tensorrt&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-142857630.142^v100^pc_search_result_base9&spm=1018.2226.3001.4187
里面的.
/home/sunh/6D_ws/Fpose_rgb/depth-anything-tensorrt-main/depth_anything_v2/engine_gen.py
生成/home/sunh/6D_ws/Fpose_rgb/feature2/depth_anything_v2_vitb.engine

然后使用/home/sunh/6D_ws/Fpose_rgb/depth-anything-tensorrt-main/my/dpt_infer.py进行预测
