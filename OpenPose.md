
/*
 *
 * WrapperStructOutput(
 * double verbose - logger
 *
 * std::string writeKeypoint - keypoint output folder ("" for disabled)
 * DataFormat writeKeypointFormat -  keypoint output format - DataFormat::Json (default), DataFormat::Xml and DataFormat::Yml
 *
 * std::string writeJson - OpenPose output in JSON format - 'people' or 'part_candidates' - ("" for disabled)
 * std::string writeCocoJson - Location of COCO  format
 *
 * int writeCocoJsonVariants - Add 1 for body, add 2 for foot, 4 for face, and/or 8 for hands. Use 0 to use all the possible candidates.
 * int writeCocoJsonVariant - experimental
 *
 * std::string writeImages - images folder ("" for disabled)
 * std::string writeImagesFormat -png, jpg etc
 *
 * std::string writeVideo; - ("" for disabled)
 *  double writeVideoFps - fps of recorded video
 * bool writeVideoWithAudio - with Audio?
 *
 *  std::string writeHeatMaps;
 * std::string writeHeatMapsFormat;
 *
 *  std::string writeVideo3D;
 * std::string writeVideoAdam;
 *
 * std::string writeBvh - ("" for disabled)
 *
 *
 *
        WrapperStructOutput(
            const double verbose = -1,
            const std::string& writeKeypoint = "",
            const DataFormat writeKeypointFormat = DataFormat::Xml,
            const std::string& writeJson = "",
            const std::string& writeCocoJson = "",
            const int writeCocoJsonVariants = 1,
            const int writeCocoJsonVariant = 1,
            const std::string& writeImages = "",
            const std::string& writeImagesFormat = "png",
            const std::string& writeVideo = "",
            const double writeVideoFps = -1.,
            const bool writeVideoWithAudio = false,
            const std::string& writeHeatMaps = "",
            const std::string& writeHeatMapsFormat = "png",
            const std::string& writeVideo3D = "",
            const std::string& writeVideoAdam = "",
            const std::string& writeBvh = "",
            const std::string& udpHost = "",
            const std::string& udpPort = "8051");
*/