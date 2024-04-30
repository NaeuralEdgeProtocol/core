AI_ENGINES = {}

AI_ENGINES['classless_detector'] = {
  'SERVING_PROCESS': 'th_obj_loc',
}

AI_ENGINES['general_detector'] = {
  # 'SERVING_PROCESS': 'th_y5l6s'
  'SERVING_PROCESS': 'th_yf8l'  # Y8L best so far for large res
}

AI_ENGINES['lowres_general_detector'] = {
  # 'SERVING_PROCESS': 'th_y5s6s'
  # 'SERVING_PROCESS': 'th_yf5s'  # y5s best so far for low res better than y8s as it has better input resolution
  'SERVING_PROCESS': 'th_yf8s'
}

AI_ENGINES['nano_general_detector'] = {
  'SERVING_PROCESS': 'th_yf8n'
}

AI_ENGINES['general_detector_xp'] = {
  'SERVING_PROCESS': 'th_yf_xp',
}

AI_ENGINES['yf5_detector'] = {
  'SERVING_PROCESS': 'th_yf5l'
}

AI_ENGINES['lowres_yf5_detector'] = {
  'SERVING_PROCESS': 'th_yf5s'
}

# AI_ENGINES['yf8_detector'] = {
#   'SERVING_PROCESS': 'th_yf8l'
# }

AI_ENGINES['lowres_yf8_detector'] = {
  'SERVING_PROCESS': 'th_yf8s'
}

AI_ENGINES['advanced_general_detector'] = {
  # 'SERVING_PROCESS': 'th_effdet7'
  'SERVING_PROCESS': 'th_yf8x'
}

AI_ENGINES['lpdr'] = {
  'SERVING_PROCESS': 'lpdr_metamodel'
}

# AI_ENGINES['plate_detector'] = {
#   'SERVING_PROCESS': 'th_y5l6s_lplr'
# }

AI_ENGINES['plate_only_detector'] = {
  'SERVING_PROCESS': 'th_lpd'
}

AI_ENGINES['plate_reader_detector'] = {
  'SERVING_PROCESS': 'th_lpd_r'
}

AI_ENGINES['plate_detector'] = {
  'SERVING_PROCESS': 'th_yf_lp'
}

AI_ENGINES['lowres_plate_detector'] = {
  'SERVING_PROCESS': 'th_yfs_lp'
}

AI_ENGINES['nano_plate_detector'] = {
  'SERVING_PROCESS': 'th_yfn_lp'
}

AI_ENGINES['image_quality_assessment_fast'] = {
  'SERVING_PROCESS': 'th_iqa_fast'
}

AI_ENGINES['image_quality_assessment_slow'] = {
  'SERVING_PROCESS': 'th_iqa_slow'
}

AI_ENGINES['image_quality_assessment'] = {
  'SERVING_PROCESS': 'th_cqc'
}

AI_ENGINES['image_quality_assessment_old'] = {
  'SERVING_PROCESS': 'th_cqc_old'
}

AI_ENGINES['covid_mask'] = {
  'SERVING_PROCESS': 'covid_mask'
}

AI_ENGINES['pose_detector'] = {
  # 'SERVING_PROCESS' : 'th_y5l6s_move',
  # 'SERVING_PROCESS': 'th_y5l6s_mspn',
  'SERVING_PROCESS': 'th_yf_mspn'
}

AI_ENGINES['lowres_pose_detector'] = {
  # 'SERVING_PROCESS': 'th_y5s6s_mspn'
  'SERVING_PROCESS': 'th_yfs_mspn'
}

AI_ENGINES['nano_pose_detector'] = {
  'SERVING_PROCESS': 'th_yfn_mspn'
}

AI_ENGINES['safety_gear_detector'] = {
  'SERVING_PROCESS': 'th_y5l6s_safety'
}

AI_ENGINES['weapon_assailant_detector'] = {
  'SERVING_PROCESS': 'th_y5l6s_weapon'
}

AI_ENGINES['face_detector'] = {
  'SERVING_PROCESS': 'th_rface'
}

AI_ENGINES['face_detector_identification'] = {
  'SERVING_PROCESS': 'th_retina_face_resnet_identification'  # 'face_identification'
}

AI_ENGINES['face_id'] = {
  'SERVING_PROCESS': 'th_rface_id'  # 'face_identification'
}

AI_ENGINES['lowres_face_detector'] = {
  'SERVING_PROCESS': 'th_rface_s'
}

AI_ENGINES['test_super_meta'] = {
  'SERVING_PROCESS': 'y5_omni',
  'PARAMS': {'OMNI_IDX': 100}
}

AI_ENGINES['custom_second_stage_detector'] = {
  # TODO: support seems to be lost downstream
  'SERVING_PROCESS': 'th_yolo_second_stage',
  'REQUIRES_INSTANCE': True  # requires ('th_yolo_second_stage', 'safety_helmet') style
}

AI_ENGINES['pose_alert_detector'] = {
  # 'SERVING_PROCESS': 'th_y5l6s_mspn_pose_clf'
  'SERVING_PROCESS': 'th_yf_mspn_pc'
}

AI_ENGINES['lowres_pose_classifier_detector'] = {
  'SERVING_PROCESS': 'th_y5s6s_mspn_pose_clf'
}

AI_ENGINES['age_gender_detector'] = {
  'SERVING_PROCESS': 'th_yf_ag'
}

AI_ENGINES['lowres_age_gender_detector'] = {
  'SERVING_PROCESS': 'th_yfs_ag'
}

AI_ENGINES['nano_age_gender_detector'] = {
  'SERVING_PROCESS': 'th_yfn_ag'
}

AI_ENGINES['planogram_stage1'] = {
  'SERVING_PROCESS': 'th_plano_s1'
}

AI_ENGINES['planogram_stage1_old'] = {
  'SERVING_PROCESS': 'th_plano_s1_old'
}
#
# AI_ENGINES['face_detector'] = {
#   'SERVING_PROCESS': 'th_face'
# }

# TODO:
#  below config is just for demo purposes
#  so this file should be split between "core" and "plugins"
AI_ENGINES['a_dummy_ai_engine'] = {
  'SERVING_PROCESS': 'a_dummy_classifier',
  'PARAMS': {
    'TEST_INFERENCE_PARAM': 1,  # should overwrite default 0
    'TEST_ENGINE_PARAM': 100,
  }
}

AI_ENGINES['a_dummy_cv_ai_engine'] = {
  'SERVING_PROCESS': 'a_dummy_cv_classifier',
  'PARAMS': {
    'TEST_INFERENCE_PARAM': 1,  # should overwrite default 0
    'TEST_ENGINE_PARAM': 100,
  }
}

AI_ENGINES['code_generator'] = {
   'SERVING_PROCESS': 'code_llama_v2'
}


AI_ENGINES['th_training'] = {
  'SERVING_PROCESS': 'th_training',
  'REQUIRES_INSTANCE': True
}
