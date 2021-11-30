import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
#config.disable_stream(rs.stream.color)
pipeline.start(config)

##Firstly align the depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)
while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
              continue
        #Get the  depth value
        depth = depth_frame.get_distance(100, 100)
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [100, 100], depth)

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
depth_value = 0.5
depth_pixel = [depth_intrin.ppx, depth_intrin.ppy]
depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)
print (depth_point)