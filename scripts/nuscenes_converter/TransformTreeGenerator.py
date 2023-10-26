#!/usr/bin/env python3
# Generate the three files ({token}.pb.txt, {token}.dag, {token}.launch) for different scenes
def GeneratePbFile(OutputPath,SceneToken):
    print(OutputPath)
    f = open(OutputPath+'static_transform_conf_'+SceneToken+'.pb.txt','w')
    # content = 'extrinsic_file {\n\tframe_id: "novatel"\n\tchild_frame_id: "velodyne32"\n\t \
    # file_path: "/apollo/modules/transform/nuScenes_Calibration/{{}}/velodyne_params/velodyne32_novatel_extrinsics.yaml"\n\t\
    # enable: true\n}\nextrinsic_file {\n\tframe_id: "localization"\n\tchild_frame_id: "novatel"\n\tfile_path: "/apollo/modules/transform/nuScenes_Calibration/novatel_localization_extrinsics.yaml"\n\t\
    # enable: true\n}\n\nextrinsic_file {\n\tframe_id: "novatel"\n\tchild_frame_id: "radar_rear_left"\n\t\
    # file_path: "/apollo/modules/transform/nuScenes_Calibration/{{}}/radar_params/radar_rear_left_extrinsics.yaml"\n\t\
    # enable: true\n}\n\nextrinsic_file {\n\tframe_id: "novatel"\n\tchild_frame_id: "radar_rear_right"\n\t\
    # file_path: "/apollo/modules/transform/nuScenes_Calibration/{{}}/radar_params/radar_rear_right_extrinsics.yaml"\n\t\
    # enable: true\n}\n\nextrinsic_file {\n\tframe_id: "novatel"\n\tchild_frame_id: "radar_front"\n\t\
    # file_path: "/apollo/modules/transform/nuScenes_Calibration/{{}}/radar_params/radar_front_extrinsics.yaml"\n\t\
    # enable: true\n}\n\nextrinsic_file {\n\tframe_id: "novatel"\n\tchild_frame_id: "radar_front_left"\n\t\
    # file_path: "/apollo/modules/transform/nuScenes_Calibration/{{}}/radar_params/radar_rear_left_extrinsics.yaml"\n\t\
    # enable: true\n}\n\nextrinsic_file {\n\tframe_id: "novatel"\n\tchild_frame_id: "radar_front_right"\n\t\
    # file_path: "/apollo/modules/transform/nuScenes_Calibration/{{}}/radar_params/radar_rear_right_extrinsics.yaml"\n\t\
    # enable: true\n}\n\n'.format(SceneToken,SceneToken)
    content = f"""
extrinsic_file {{
    frame_id: "novatel"
    child_frame_id: "velodyne32"
    file_path: "/apollo/modules/transform/nuScenes_Calibration/{SceneToken}/velodyne_params/velodyne32_novatel_extrinsics.yaml"
    enable: true
}}
    
extrinsic_file {{
    frame_id: "localization"
    child_frame_id: "novatel"
    file_path: "/apollo/modules/transform/nuScenes_Calibration/novatel_localization_extrinsics.yaml"
    enable: true
}}

extrinsic_file {{
    frame_id: "novatel"
    child_frame_id: "radar_rear_left"
    file_path: "/apollo/modules/transform/nuScenes_Calibration/{SceneToken}/radar_params/radar_rear_left_extrinsics.yaml"
    enable: true
}}
extrinsic_file {{
    frame_id: "novatel"
    child_frame_id: "radar_rear_right"
    file_path: "/apollo/modules/transform/nuScenes_Calibration/{SceneToken}/radar_params/radar_rear_right_extrinsics.yaml"        
    enable: true
}}

extrinsic_file {{
    frame_id: "novatel"
    child_frame_id: "radar_front"
    file_path: "/apollo/modules/transform/nuScenes_Calibration/{SceneToken}/radar_params/radar_front_extrinsics.yaml"
    enable: true
}}
extrinsic_file {{
    frame_id: "novatel"
    child_frame_id: "radar_front_left"
    file_path: "/apollo/modules/transform/nuScenes_Calibration/{SceneToken}/radar_params/radar_front_left_extrinsics.yaml"
    enable: true
}}

extrinsic_file {{
    frame_id: "novatel"
    child_frame_id: "radar_front_right"
    file_path: "/apollo/modules/transform/nuScenes_Calibration/{SceneToken}/radar_params/radar_rear_right_extrinsics.yaml"
    enable: true
}}
    """
    f.write(content)
    f.close()

    return
    

def GenerateDagFile(OutputPath,SceneToken):
    f = open(OutputPath+'static_transform_'+SceneToken+'.dag','w')
    # content = '# Define all coms in DAG streaming.\n\
    #            module_config {\n\t\
    #            module_library : "/apollo/bazel-bin/modules/transform/libstatic_transform_component.so"\n\t\
    #            components {\n\t\t\
    #            class_name : "StaticTransformComponent"\n\t\t\
    #            config {\n\t\t\t\
    #            name : "static_transform"\n\t\t\t\
    #            config_file_path: "/apollo/modules/transform/conf/static_transform_conf_{}.pb.txt"\n\t\t\
    #            }\n\t}\n}'.format(SceneToken)
    content = f"""
# Define all coms in DAG streaming.
module_config {{
    module_library : "/apollo/bazel-bin/modules/transform/libstatic_transform_component.so"
    components {{
        class_name : "StaticTransformComponent"
        config {{
            name : "static_transform"
            config_file_path: "/apollo/modules/transform/conf/static_transform_conf_{SceneToken}.pb.txt"
        }}
    }}
}}
"""
    f.write(content)
    f.close()
    return

def GenerateLaunchFile(OutputPath,SceneToken):
    f = open(OutputPath+'static_transform_'+SceneToken+'.launch','w')
    # content = '<cyber>\n\t\
    #            <module>\n\t\t\
    #            <name>static_transform</name>\n\t\t\
    #            <dag_conf>/apollo/modules/transform/dag/static_transform_{}.dag</dag_conf>\n\t\t\
    #            <process_name></process_name>\n\t\
    #            </module>\n\t\
    #            </cyber>'.format(SceneToken)
    content = '''
<cyber>
    <module>
        <name>static_transform</name>
        <dag_conf>/apollo/modules/transform/dag/static_transform_{}.dag</dag_conf>
        <process_name></process_name>
    </module>
</cyber>
'''.format(SceneToken)
    f.write(content)
    f.close()
    return

def TransformTreeGenerator(OutputPath,SceneToken):
    # Given the path of transform module, usually it is modules/transfrom
    GeneratePbFile(OutputPath+'conf/',SceneToken)
    GenerateDagFile(OutputPath+'dag/',SceneToken)
    GenerateLaunchFile(OutputPath+'launch/',SceneToken)
    return