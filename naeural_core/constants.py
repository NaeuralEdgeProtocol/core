from PyE2.const import (
  PAYLOAD_CT,
  COMMANDS,
  STATUS_TYPE,
  PAYLOAD_DATA,
  CONFIG_STREAM,
  BIZ_PLUGIN_DATA,
  PLUGIN_INFO,
  COLORS,
  HB,
  BASE_CT,
  COMMS,
  NOTIFICATION_CODES,
  WEEKDAYS_SHORT,
)

BLOCKCHAIN_MANAGER = 'BLOCKCHAIN_MANAGER'

WHITELIST_COMMANDS_FILE = 'whitelist_commands.json'
TEMPLATE_WHITELIST_COMMANDS = [
  # TODO: add other commands
  # !! NetMon queries
  {
    "ACTION": COMMANDS.UPDATE_PIPELINE_INSTANCE,
    "PAYLOAD": {
      "SIGNATURE": "NET_MON_01",
      "INSTANCE_ID": "NET_MON_01_INST",
      "INSTANCE_CONFIG": {
        "INSTANCE_COMMAND": {
        }
      }
    }
  },
  
  # !! Get config startup
  {
    "ACTION": COMMANDS.UPDATE_PIPELINE_INSTANCE,
    "PAYLOAD": {
      "SIGNATURE": "UPDATE_MONITOR_01",
      "INSTANCE_CONFIG": {
        "INSTANCE_COMMAND": {
          "COMMAND": "GET_CONFIG",
        }
      }
    },
  },
  # !! Edit config startup - requires authorization
  # {
  #   "ACTION": COMMANDS.UPDATE_PIPELINE_INSTANCE,
  #   "PAYLOAD": {
  #     "SIGNATURE": "UPDATE_MONITOR_01",
  #     "INSTANCE_CONFIG": {
  #       "INSTANCE_COMMAND": {
  #         "COMMAND": "SAVE_CONFIG",
  #       }
  #     }
  #   },
  # },

  # !! Restart Edge Node - requires authorization
  # {
  #   "ACTION": COMMANDS.STOP,
  # },

  # !! Restart OS - requires authorization
  # {
  #   "ACTION": COMMANDS.RESTART,
  # },

  # Request heartbeat with extra info
  {
    "ACTION": COMMANDS.FULL_HEARTBEAT,
  },
  {
    "ACTION": COMMANDS.TIMERS_ONLY_HEARTBEAT,
  },
]

ADMIN_PIPELINE = {
  "PLUGIN_MONITOR_01":{
    "PROCESS_DELAY": 1,
  },

  "LOCAL_COMMS_01": {
    "LOCAL_COMMS_ENABLED_ON_STARTUP": False
  },

  "MINIO_MONIT_01": {
    "PROCESS_DELAY": 20,
    "MINIO_HOST": None,
    "MINIO_ACCESS_KEY": None,
    "MINIO_SECRET_KEY": None,
    "MINIO_SECURE": None,
    "MAX_SERVER_QUOTA"  : 95
  },

  "REST_CUSTOM_EXEC_01": {
    "ALLOW_EMPTY_INPUTS": True,
    "RUN_WITHOUT_IMAGE": True,
    "SEND_MANIFEST_EACH": 301
  },

  "SELF_CHECK_01": {
    "DISK_LOW_PRC": 0.15,
    "MEM_LOW_PRC": 0.15,
    "PROCESS_DELAY": 20
  },

  "NET_MON_01": {
    "PROCESS_DELAY": 20,
    "SUPERVISOR": False
  },

  "UPDATE_MONITOR_01": {
    "PROCESS_DELAY": 120,
    "WORKING_HOURS": [["08:30", "09:30"]],
    "VERSION_TOKEN": None,
    "RESTART_ON_BEHIND": True,
    "VERSION_URL": "https://raw.githubusercontent.com/NaeuralEdgeProtocol/edge_node/ver.py"
  },
  
  "K8S_MONITOR_01": {
    "PROCESS_DELAY": 10,
  },

  "SYSTEM_HEALTH_MONITOR_01": {
    "PROCESS_DELAY": 180,
    "KERNEL_LOG_LEVEL" : "emerg,alert,crit,err"
  }
}

ADMIN_PIPELINE_NETMON = 'NET_MON_01'

NOTIFICATION_TYPE = STATUS_TYPE

NOTIFICATION = PAYLOAD_DATA.NOTIFICATION

EE_ID = BASE_CT.EE_ID
EE_ADDR = HB.EE_ADDR

NON_ACTIONABLE_INSTANCE_CONFIG_KEYS = [
  'INSTANCE_COMMAND_LAST',
]


LOCAL_CACHE = '_local_cache'

CONFIG_APP_FALLBACK = 'CONFIG_APP_FALLBACK'
CONFIG_APP_DOCKER_FALLBACK = './scripts/docker_env_ext/config_app.txt'

# CV payload
TLBR_POS = PAYLOAD_CT.TLBR_POS
PROB_PRC = PAYLOAD_CT.PROB_PRC
TYPE = PAYLOAD_CT.TYPE


PLUGIN_END_PREFIX = ">>>> STOP: "
BM_PLUGIN_END_PREFIX = ">> STOP: "


class CALLBACKS:
  INSTANCE_CONFIG_SAVER_CALLBACK = 'instance_config_saver_callback'
  PIPELINE_CONFIG_SAVER_CALLBACK = 'pipeline_config_saver_callback'
  MAIN_LOOP_RESOLUTION_CALLBACK = 'main_loop_resolution_callback'


class IPFS:
  UPLOAD_CONFIG = 'UPLOAD_CONFIG'


class SERVING:
  SERVER_COLLECTOR_TIMEDELTA = 'SERVER_COLLECTOR_TIMEDELTA'
  SERVER_COLLECTOR_DELAY = 'SERVER_COLLECTOR_DELAY'
  CHECK_BLOCKED_INPROCESS_SERVING = 'CHECK_BLOCKED_INPROCESS_SERVING'
  MAX_WAIT_TIME_MULTIPLIER = 'MAX_WAIT_TIME_MULTIPLIER'
  SHM_IMG_MAX_SHAPE = (1520, 2688, 3)
  SHM_MAX_LEN = 50
  COMM_METHOD = 'COMM_METHOD'


class CONFIG_APP_v2:
  K_COMMUNICATION = 'COMMUNICATION'


class CONFIG_MANAGER:
  DEFAULT_SUBFOLDER_PATH = 'box_configuration'
  DEFAULT_FN_APP_CONFIG = 'config_app.txt'
  DEFAULT_FOLDER_STREAMS_CONFIGS = 'streams'


class CONFIG_INSTANCE:
  K_INSTANCE_ID = BIZ_PLUGIN_DATA.INSTANCE_ID


class CONFIG_INSTANCE_COMPUTER_VISION(CONFIG_INSTANCE):
  K_COORDS = BIZ_PLUGIN_DATA.COORDS
  K_POINTS = BIZ_PLUGIN_DATA.POINTS


class CONFIG_PLUGIN:
  K_INSTANCES = BIZ_PLUGIN_DATA.INSTANCES
  K_SIGNATURE = BIZ_PLUGIN_DATA.SIGNATURE


class CONFIG_RETRIEVE:
  K_TYPE = 'TYPE'
  K_CONNECT_PARAMS = 'CONNECT_PARAMS'
  K_APP_CONFIG_ENDPOINT = 'APP_CONFIG_ENDPOINT'
  K_STREAMS_CONFIGS_ENDPOINT = 'STREAMS_CONFIGS_ENDPOINT'


class CONFIG_STARTUP_v2:
  K_EE_ID = 'EE_ID'
  K_LITE = 'LITE'
  K_CONFIG_RETRIEVE = 'CONFIG_RETRIEVE'
  DEFAULT_EE_ID = 'XXXXXXXXXX'
  K_SECURED = 'SECURED'
  SECURED = K_SECURED
  BLOCKCHAIN_CONFIG = 'BLOCKCHAIN_CONFIG'
  CHECK_RAM_ON_SHUTDOWN = 'CHECK_RAM_ON_SHUTDOWN'


class PLUGIN_SEARCH:
  # SEARCH_PACKAGES
  SEARCH_IN_PACKAGES = ["PyE2", "naeural_core"]

  # CONFIG
  LOC_CONFIG_RETRIEVE_PLUGINS = 'plugins.config'
  SUFFIX_CONFIG_RETRIEVE_PLUGINS = 'ConfigRetriever'
  SAFE_LOC_CONFIG_PLUGINS = ['naeural_core.config.default']

  # COMM
  LOC_COMM_PLUGINS = 'plugins.comm'
  SUFFIX_COMM_PLUGINS = 'CommThread'
  SAFE_LOC_COMM_PLUGINS = ['naeural_core.comm.default',]

  # DATA
  LOC_DATA_ACQUISITION_PLUGINS = ['plugins.data', 'extensions.data']
  SAFE_LOC_DATA_ACQUISITION_PLUGINS = ['naeural_core.data.default', 'naeural_core.data.training', 'extensions.data']
  SUFFIX_DATA_ACQUISITION_PLUGINS = 'DataCapture'
  SAFE_LOC_DATA_ACQUISITION_IMPORTS = ['naeural_core.data', 'extensions.data']

  # SERVING
  LOC_SERVING_PLUGINS = ['plugins.serving.inference', 'plugins.serving.training']
  SUFFIX_SERVING_PLUGINS = None
  SAFE_LOC_SERVING_PLUGINS = ['naeural_core.serving.default_inference', 'naeural_core.serving.training', 'extensions.serving']
  SERVING_SAFE_IMPORTS = ['naeural_core.serving', 'extensions.serving']  # be aware that someone may drop some .py there and "validate" attacker code

  # BUSINESS
  LOC_BIZ_PLUGINS = ['plugins.business']
  SAFE_BIZ_PLUGINS = ['naeural_core.business.default', 'naeural_core.business.training', 'extensions.business']  # this is were the safe plugins are stored
  SAFE_BIZ_IMPORTS = ['naeural_core.business', 'extensions.business']  # be aware that someone may drop some .py there and "validate" attacker code
  SUFFIX_BIZ_PLUGINS = 'Plugin'

  # Heavy Ops
  SAFE_LOC_HEAVY_OPS_PLUGINS = 'naeural_core.heavy_ops.default'
  LOC_HEAVY_OPS_PLUGINS = 'plugins.heavy_ops'
  SUFFIX_HEAVY_OPS_PLUGINS = 'HeavyOp'

  # FILE SYS
  LOC_FILE_SYSTEM_PLUGINS = 'plugins.remote_file_system'
  SUFFIX_FILE_SYSTEM_PLUGINS = 'FileSystem'
  LOC_SAFE_FILE_SYSTEM_PLUGINS = ['naeural_core.remote_file_system.default']

  # TESTING FRAMEWORK
  LOC_TESTING_FRAMEWORK_TESTING_PLUGINS = 'plugins.business.test_framework.testing'
  SUFFIX_TESTING_FRAMEWORK_TESTING_PLUGINS = 'TestingPlugin'

  LOC_TESTING_FRAMEWORK_SCORING_PLUGINS = 'plugins.business.test_framework.scoring'
  SUFFIX_TESTING_FRAMEWORK_SCORING_PLUGINS = 'ScoringPlugin'


class RETURN_CODES:
  CODE_RESTART = 10
  CODE_EXCEPTION = 11
  CODE_STOP = 12


class SYS_MON:
  NAME = 'SYSTEM_MONITOR'


class EMAIL_NOTIFICATION:
  EMAIL_CONFIG = 'EMAIL_CONFIG'
  DEFAULT_EMAIL_CONFIG = None
  SERVER = 'SERVER'
  DEFAULT_SERVER = ''
  PORT = 'PORT'
  DEFAULT_PORT = 25
  USER = 'USER'
  DEFAULT_USER = 'tmp@email.com'
  PASSWORD = 'PASSWORD'
  DEFAULT_PASSWORD = ''
  DESTINATION = 'DESTINATION'


class IMAGE_COMPRESSION:
  MAX_HEIGHT = 720
  QUALITY = 60
  ORIG_MAX_HEIGHT = 1080


class MINIO:
  ACCESS_KEY = 'access_key'
  SECRET_KEY = 'secret_key'
  BUCKET_NAME = 'bucket_name'
  ENDPOINT = 'endpoint'
  FILE_RETENTION = 'file_retention'
  SECURE = 'secure'


class SHMEM:
  BUSINESS = 'BUSINESS'
  SERVING = 'SERVING'
  DCT = 'DCT'
  COMM = 'COMM'


class GPU_INFO:
  INFO = 'GPU_INFO'
  NAME = 'NAME'
  TOTAL_MEM = 'TOTAL_MEM'
  GPU_USED = 'GPU_USED'
  ALLOCATED_MEM = 'ALLOCATED_MEM'
  GPU_TEMP = "GPU_TEMP"
  GPU_FAN_SPEED = "GPU_FAN_SPEED"
  GPU_FAN_SPEED_UNIT = "GPU_FAN_SPEED_UNIT"


class MOVIE_EVAL_STAGES:
  INFERENCE = 'INFERENCE'
  WRITE = 'WRITE'


class FILE_SIZE_UNIT:
  BYTES = 'BYTES'
  KB = 'KB'
  MB = 'MB'
  GB = 'GB'


CUSTOM_DOWNLOADABLE_MODEL = 'CUSTOM_DOWNLOADABLE_MODEL'
CUSTOM_DOWNLOADABLE_MODEL_URL = 'CUSTOM_DOWNLOADABLE_MODEL_URL'
MODEL_ZOO_CONFIG = 'MODEL_ZOO_CONFIG'
SERVING_TIMERS_IDLE_DUMP = 'SERVING_TIMERS_IDLE_DUMP'
SERVING_TIMERS_PREDICT_DUMP = 'SERVING_TIMERS_PREDICT_DUMP'

# root engine version
LITE = 'LITE'


# communication
COMMUNICATORS = 'COMMUNICATORS'
COMMUNICATOR_REST = 'REST'
COMMUNICATOR_PIKA = 'PIKA'
COMMUNICATOR_PAHO = 'PAHO'

CONN_MAX_RETRY_ITERS = 'CONN_MAX_RETRY_ITERS'

AVAILABLE_STREAMS_PICTURES = 'AVAILABLE_STREAMS_PICTURES'
CHECK = 'CHECK'
EXCEPTION = 'EXCEPTION'
HEARTBEAT = 'HEARTBEAT'
PAYLOAD = 'PAYLOAD'
RESTART = 'RESTART'
STATUS = 'STATUS'
STOP = 'STOP'
UPDATE_CONFIG = 'UPDATE_CONFIG'
DEVICE_STATUS = 'DEVICE_STATUS'
DEVICE_MESSAGE = 'DEVICE_MESSAGE'

DEVICE_LOG = 'DEVICE_LOG'
ERROR_LOG = 'ERROR_LOG'

# DEVICE STATUS
DEVICE_STATUS_AVAILABLE_STREAMS_PICTURES = 'AVAILABLE_STREAMS_PICTURES'
DEVICE_STATUS_CHECK = 'CHECK'
DEVICE_STATUS_EXCEPTION = 'EXCEPTION'
DEVICE_STATUS_ONLINE = 'ONLINE'
DEVICE_STATUS_RESTART = 'RESTART'
DEVICE_STATUS_SHUTDOWN = 'SHUTDOWN'
DEVICE_STATUS_STOP = 'STOP'
DEVICE_STATUS_UPDATE_CONFIG = 'UPDATE_CONFIG'

ID = 'ID'
ACTION = 'ACTION'

MQ_BROKER = 'MQ_BROKER'
MQ_PORT = 'MQ_PORT'
MQ_USER = 'MQ_USER'
MQ_PASS = 'MQ_PASS'
MQ_PATH = 'MQ_PATH'

MQ_CONFIG_EXCHANGE = 'CONFIG_EXCHANGE'
MQ_CONFIG_EXCHANGE_TYPE = 'CONFIG_EXCHANGE_TYPE'
MQ_CONFIG_QUEUE = 'CONFIG_QUEUE'
MQ_CONFIG_ROUTING_KEY = 'CONFIG_ROUTING_KEY'
MQ_CONFIG_QUEUE_DURABLE = 'CONFIG_QUEUE_DURABLE'
MQ_CONFIG_QUEUE_EXCLUSIVE = 'CONFIG_QUEUE_EXCLUSIVE'
MQ_CONFIG_QUEUE_DEVICE_SPECIFIC = 'CONFIG_QUEUE_DEVICE_SPECIFIC'


DISABLE_THREAD = 'DISABLE_THREAD'


MQ_PAYLOADS_EXCHANGE = 'PAYLOADS_EXCHANGE'
MQ_PAYLOADS_EXCHANGE_TYPE = 'PAYLOADS_EXCHANGE_TYPE'
MQ_PAYLOADS_QUEUE = 'PAYLOADS_QUEUE'
MQ_PAYLOADS_ROUTING_KEY = 'PAYLOADS_ROUTING_KEY'
MQ_PAYLOADS_QUEUE_DURABLE = 'PAYLOADS_QUEUE_DURABLE'
MQ_PAYLOADS_QUEUE_EXCLUSIVE = 'PAYLOADS_QUEUE_EXCLUSIVE'
MQ_PAYLOADS_QUEUE_DEVICE_SPECIFIC = 'PAYLOADS_QUEUE_DEVICE_SPECIFIC'

COMMUNICATION = 'COMMUNICATION'
NR_PAYLOADS_TO_SEND = 'NR_PAYLOADS_TO_SEND'
BUFFER_MAX_LEN = 'BUFFER_MAX_LEN'


# config KEYS
CONFIG_APP = 'CONFIG_APP'
CONFIG_PLUGINS = 'CONFIG_PLUGINS'
CONFIG_INFERENCE = 'CONFIG_INFERENCE'
CONFIG_INFERENCE_BENCHMARK = 'CONFIG_INFERENCE_BENCHMARK'
CONFIG_STREAMS = 'CONFIG_STREAMS'
CONFIG_DEBUG_WINDOW = 'CONFIG_DEBUG_WINDOW'
CONFIG_STARTUP = 'CONFIG_STARTUP'
CONFIG_MODELS_ZOO = 'CONFIG_MODELS_ZOO'
CONFIG_MODELS_ZOO_BENCHMARK = 'CONFIG_MODELS_ZOO_BENCHMARK'


# RETURN CODES
CODE_STOP = 0
CORE_NODE_RESTART = CODE_STOP
CODE_SHUTDOWN = 1
CODE_RESTART = 10
CODE_BOX_RESTART = CODE_RESTART
CODE_EXCEPTION = 51
CODE_CONFIG_ERROR = 52


# STREAMS STATES
STATE_ALIVE = 'ALIVE'
STATE_EXCEPTION = 'EXCEPTION'
STATE_FINISHED = 'FINISHED'
STATE_STOPPED = 'STOPPED'
STATE_KILLED = 'KILLED'

# STATUS TYPES
STATUS_NORMAL = 'NORMAL'
STATUS_EXCEPTION = 'EXCEPTION'

# TIMERS
TIMER_APP = 'app_alive_timer'
TIMER_ASCONTIGUOUSARRAY = 'ascontiguousarray'
TIMER_BGR_WITNESS = 'bgr_witness'
TIMER_CA_ANCHOR = 'ca_anchor'
TIMER_CA_ANCHOR_MEAN = 'ca_anchor_mean'
TIMER_CA_COLLECT_ANCHOR = 'ca_collect_anchor'
TIMER_CA_COLLECT_CURRENT = 'ca_collect_current'
TIMER_CA_CREATE_ANCHOR_MEAN = 'ca_create_anchor_mean'
TIMER_CA_CREATE_ANCHOR = 'ca_create_anchor'
TIMER_CA_CREATE_CURRENT = 'ca_create_current'
TIMER_CA_CREATE_CURRENT_MEAN = 'ca_create_current_mean'
TIMER_CA_EVALUATE = 'ca_evaluate'
TIMER_CA_SSIM = 'ca_ssim'
TIMER_CA_SSIM_OPERATION = 'ca_ssim_operation'
TIMER_CA_GET_SURF_FEATURES = 'ca_surf'
TIMER_CA_MATCH_KEYPOINTS = 'ca_match_kpts'
TIMER_CA_HOMOGRAPHY = 'ca_homography'
TIMER_CA_GET_DISTANCE_PLUS_LABELS = 'ca_eval_dist_lbl'
TIMER_CA_GET_DISTANCE = 'ca_eval_dist'
TIMER_COPY_WITNESS = 'copy_witness'
TIMER_CQC = 'cqc'
TIMER_CQC_COLLECT = 'cqc_collect'
TIMER_CQC_EVALUATE = 'cqc_evaluate'
TIMER_GET_CAPTURES_DATA = 'get_captures_data'
TIMER_GET_CAPTURES_MESSAGES = 'get_captures_messages'
TIMER_GET_WITNESS_IMAGE = 'get_witness_image'
TIMER_INFERENCE = 'inference'
TIMER_INFER = 'infer'
TIMER_MAIN_LOOP = 'main_loop'
TIMER_PREPARE_PAYLOAD = 'prepare_payload'
TIMER_PREPARE_RESULTS = 'prepare_results'
TIMER_PRE_PROCESS = 'TIMER_PRE_PROCESS'
TIMER_PREPROCESS_WITNESS = 'pre_process_witness'
TIMER_POSTPROCESS_WITNESS = 'post_process_witness'
TIMER_REFRESH_APP = 'refresh_app'
TIMER_REFRESH_PLUGINS = 'refresh_plugins'
TIMER_REFRESH_STREAMS = 'refresh_streams'
TIMER_RUN_PLUGINS = 'run_plugins'
TIMER_SAVE_BUFFER_TO_DISK = 'save_buffer_to_disk'
TIMER_SEND_BUFFER_PAYLOAD = 'send_buffer_payload'
TIMER_SEND_INSTANT_PAYLOAD = 'send_instant_payload'
TIMER_SEND_PRIORITY_PAYLOAD = 'send_priority_payload'
TIMER_SET_STREAM_CONFIG = 'set_stream_config'
TIMER_WITNESS_BASE64 = 'witness_base64'


# COMMON STRINGS
ACTION_ADD = 'ADD'
ACTION_DELETE = 'DELETE'
ACTION_UPDATE = 'UPDATE'
ACCESS_TOKEN = 'ACCESS_TOKEN'
ACCESS_ALLOWED = 'ACCESS_ALLOWED'
ACCESS_ALLOWED_TIME = 'ACCESS_ALLOWED_TIME'
ACCESS_ALLOWED_NOTIFIED = 'ACCESS_ALLOWED_NOTIFIED'
ACTIVE_PLUGINS = 'ACTIVE_PLUGINS'
AI_ENGINE = 'AI_ENGINE'
ALERT_TYPE = 'ALERT_TYPE'
ALERT_THRESHOLD = 'ALERT_THRESHOLD'
ALERT_MODE = 'ALERT_MODE'
ALERT_RAISE_VALUE = 'ALERT_RAISE_VALUE'
ALERT_LOWER_VALUE = 'ALERT_LOWER_VALUE'
ALERT_HELPER = 'ALERT_HELPER'
ALIVE_TIME_MINS = 'ALIVE_TIME_MINS'
ALLOCATED_MEM = 'ALLOCATED_MEM'
ALERT_OBJECTS = 'ALERT_OBJECTS'


ANOMALY_PROBA = 'ANOMALY_PROBA'
ANOMALY_THRESHOLD = 'ANOMALY_THRESHOLD'
ANGLE = 'ANGLE'
ANGLE_LOW = 'ANGLE_LOW'
ANGLE_HIGH = 'ANGLE_HIGH'
ANGLE_ANCHORS = 'ANGLE_ANCHORS'
ANGLE_RESIZE = 'ANGLE_RESIZE'
ANGLE_SSIM_INTERVALS = 'ANGLE_SSIM_INTERVALS'
ANGLE_THRESHOLD = 'ANGLE_THRESHOLD'
APPEARANCES = 'APPEARANCES'
APPLICATION = 'APPLICATION'
APPLICATION_STATUS = 'APPLICATION_STATUS'
APPMON_VERSION = 'APPMON_VERSION'
ARRIVAL_TIME = 'ARRIVAL_TIME'
AVAILABLE_MEMORY = 'AVAILABLE_MEMORY'
AVAILABLE_DISK = 'AVAILABLE_DISK'
AVERAGE = 'AVERAGE'
AUDIT_PLUGINS = 'audit_plugins'
BASIC_OBSTRUCTION_MIN_CONFIDENCE = 'BASIC_OBSTRUCTION_MIN_CONFIDENCE'
BATCH_STRATEGY = 'BATCH_STRATEGY'
BATCH_STRATEGY_MOST_COMMON_SHAPE = 'BATCH_STRATEGY_MOST_COMMON_SHAPE'
BG_SUBTRACTOR_VAR_THRS = 'BG_SUBTRACTOR_VAR_THRS'
BIG_CIRCLE_RADIUS_PERCENT = 'BIG_CIRCLE_RADIUS_PERCENT'
BLUR = 'BLUR'
BLUR_ADAPTIVE = 'BLUR_ADAPTIVE'
BLUR_ON = 'BLUR_ON'
BLUR_RAW = 'BLUR_RAW'
BLUR_ZONES = 'BLUR_ZONES'
BOTH = 'BOTH'
BOTTOM = 'BOTTOM'
BOX_OFFSET = 'BOX_OFFSET'
CA_ALERT = 'CA_ALERT'
CA_ANCHOR = 'CA_ANCHOR'
CA_ANCHORS = 'CA_ANCHORS'
CA_ANCHOR_SOURCE = 'CA_ANCHOR_SOURCE'
CA_CURRENT = 'CA_CURRENT'
CA_CURRENT_SOURCE = 'CA_CURRENT_SOURCE'
CA_IMAGE = 'CA_IMAGE'
CA_STATE = 'CA_STATE'
CA_SSIM = 'CA_SSIM'
CA_SSIM_LABEL = 'CA_SSIM_LABEL'
CA_DIST = 'CA_DIST'
CA_DIST_LABEL = 'CA_DIST_LABEL'
CACHE = 'CACHE'

CACHE_DATA_CAPTURE = 'cache_data_capture'
CACHE_PLUGINS = 'cache_plugins'
CACHE_SERVING = 'cache_serving'

CLASSES = 'CLASSES'
CLOSE = 'CLOSE'
COLOR = 'COLOR'
COLOR_RESOLUTION_INTERVAL = 'COLOR_RESOLUTION_INTERVAL'
COLLECTOR = 'COLLECTOR'
COMM_TYPE = 'COMM_TYPE'
CONFIG = 'CONFIG'
CONFIDENCE_THRESHOLD = 'CONFIDENCE_THRESHOLD'
COORDONATES = 'COORDONATES'
COUNT = 'COUNT'
CPU = 'CPU'
COVERED_SERVERS = 'COVERED_SERVERS'
CPU_USED = 'CPU_USED'
CQC_ALERT = 'CQC_ALERT'  # check if in use
CQC_CLASSES = 'CQC_CLASSES'  # check if in use
CQC_DATA = 'CQC_DATA'  # check if in use
CQC_INFERENCE = 'CQC_INFERENCE'  # check if in use
CQC_LABEL = 'CQC_LABEL'  # check if in use
CQC_PROC = 'CQC_PROC'  # check if in use
CQC_PROBS = 'CQC_PROBS'  # check if in use
CQC_OBSTRUCTION = 'OBSTRUCTION'


CURRENT_TIME = 'CURRENT_TIME'
COMMUNICATION_THREAD = 'Thread'
COMMAND = 'COMMAND'
CONFIG_RETRIEVER = 'ConfigRetriever'
COORDS = 'COORDS'
COORDS_NONE = 'NONE'
COORDS_POINTS = 'POINTS'
COORDS_TLBR = 'TLBR'
LOOPS_TIMINGS = 'LOOPS_TIMINGS'
DAILY_INTERVALS = 'DAILY_INTERVALS'
DATA = 'DATA'
DATA_ACQUISITION = 'DATA_ACQUISITION'
DATA_HELPER_NAME = 'DATA_HELPER_NAME'
DATA_HELPER_PARAMS = 'DATA_HELPER_PARAMS'
DATA_CAPTURE_THREAD = 'DataCaptureThread'
DATA_CAPTURE = 'DataCapture'
DATE = 'DATE'
DEFAULT_EE_ID = 'XXXXXXXXXX'
DISABLE = 'DISABLE'
DEBUG = 'DEBUG'
DEBUG_DATA = 'DEBUG_DATA'
DEBUG_INFO = 'DEBUG_INFO'
DEBUG_PROP = 'DEBUG_PROP'
DEBUG_META = 'DEBUG_META'
DEBUG_TIMERS = 'DEBUG_TIMERS'
DEBUG_WINDOW = 'DEBUG_WINDOW'
DEBUG_OBJECTS = 'DEBUG_OBJECTS'
DEBUG_OBJECTS_SUMMARY = 'DEBUG_OBJECTS_SUMMARY'
DEBUG_PAYLOADS = 'DEBUG_PAYLOADS'
DEFAULT_CUDA = 'DEFAULT_DEVICE'
DEMO_MODE = 'DEMO_MODE'
DESCRIPTION = 'DESCRIPTION'
DETECTIONS = 'DETECTIONS'
DEVICE_ID = 'DEVICE_ID'
DOWNLOAD = 'DOWNLOAD'
DOWNLOAD_ELAPSED_TIME = 'DOWNLOAD_ELAPSED_TIME'
DOWNLOAD_PROGRESS = 'DOWNLOAD_PROGRESS'
DOWNLOAD_START_DATE = 'DOWNLOAD_START_DATE'
DOWNLOADS = 'DOWNLOADS'
DRAW_CACHE = 'DRAW_CACHE'
DROPBOX = 'DROPBOX'
ELAPSED_TIME = 'ELAPSED_TIME'
ENABLE = 'ENABLE'
ENABLE_QUALITY_CHECK = 'ENABLE_QUALITY_CHECK'
ENABLE_CAMERA_ANGLE = 'ENABLE_CAMERA_ANGLE'
ENABLE_CAMERA_OBSTRUCTION = 'ENABLE_CAMERA_OBSTRUCTION'
ENABLE_HEURISTIC_CAMERA_OBSTRUCTION = 'ENABLE_HEURISTIC_CAMERA_OBSTRUCTION'
ENABLE_MODEL_BASED_CAMERA_OBSTRUCTION = 'ENABLE_MODEL_BASED_CAMERA_OBSTRUCTION'
ENABLED = 'ENABLED'
END = 'END'
ENTROPY_DISK_RADIUS = 'ENTROPY_DISK_RADIUS'
ENTROPY_RATIO_BREAKPOINT = 'ENTROPY_RATIO_BREAKPOINT'
ERROR = 'ERROR'
EXCLUDE_ZONES = 'EXCLUDE_ZONES'
EXP_DT = 'EXP_DT'
EXP_ST = 'EXP_ST'
EXPIRATION_SECONDS = 'EXPIRATION_SECONDS'
EXCEEDING_STANDING_STILL = 'EXCEEDING_STANDING_STILL'
EXCEEDING_STANDING_STILL_TIME = 'EXCEEDING_STANDING_STILL_TIME'
EXCEEDING_STANDING_STILL_NOTIFIED = 'EXCEEDING_STANDING_STILL_NOTIFIED'
EXCLUDE_OBJECTS = 'EXCLUDE_OBJECTS'
FA_FNA = 'FA_FNA'
FILE_UPLOAD = 'FILE_UPLOAD'
FIXED_ZONES = 'FIXED_ZONES'
FOLDER_DATA = 'data'
FOLDER_MODELS = 'models'
FOLDER_NAME = 'FOLDER_NAME'
FOLDER_OUTPUT = 'output'
FOLDER_LOGS = 'logs'
FOLDER_DOWNLOADS = 'downloads'
FORCE_UPDATE = 'FORCE_UPDATE'
FPS = 'FPS'
FPS_OUTPUT = 'FPS_OUTPUT'
FRAME_H = 'FRAME_H'
FRAME_W = 'FRAME_W'
FRAME_CURRENT = 'FRAME_CURRENT'
FRAME_COUNT = 'FRAME_COUNT'
FRAME_INTERVAL = 'FRAME_INTERVAL'
FRAMES_INTERVAL = 'FRAMES_INTERVAL'
FRAMEWORK = 'FRAMEWORK'
FULL = 'FULL'
FULL_BLUR = 'FULL_BLUR'
GIT_BRANCH = 'GIT_BRANCH'
GPUS = 'GPUS'
GPU_MIN_MEMORY_MB = 'GPU_MIN_MEMORY_MB'
GOOD = 'GOOD'
GRAPH = 'GRAPH'
GRAPH_TYPE = 'GRAPH_TYPE'
GQL = 'GQL'
HAS_FINISHED = 'HAS_FINISHED'
HAS_PAYLOAD = 'HAS_PAYLOAD'
HEATMAP_RECORD_INTERVAL = 'HEATMAP_RECORD_INTERVAL'
HISTORY = 'HISTORY'
HOST = 'HOST'

IMG = 'IMG'
IMAGE = 'IMAGE'
IMAGE_HW = 'IMAGE_HW'
INSTANCES = 'INSTANCES'
IN_CONFIRMATION = 'IN_CONFIRMATION'
INCREASE_WITH_PIXELS = 'INCREASE_WITH_PIXELS'
INFER = 'INFER'
INFERENCE = 'INFERENCE'
INSTANCE_INFERENCES = 'INSTANCE_INFERENCES'
INFERENCE_EXECUTOR = 'INFERENCE_EXECUTOR'
INFERENCE_GRAPHS = 'INFERENCE_GRAPHS'
INFERENCE_GRAPHS_OUTPUT = 'INFERENCE_GRAPHS_OUTPUT'
INFERENCE_META_GRAPHS = 'INFERENCE_META_GRAPHS'
INFERENCE_ON_FULL_IMAGE = 'INFERENCE_ON_FULL_IMAGE'
INFERENCE_TS = 'INFERENCE_TS'
INFERENCE_GRAPHS = 'INFERENCE_GRAPHS'
INFER_INTERSECT = 'INFER_INTERSECT'
INSTANCES = 'INSTANCES'
INSTANCE = 'INSTANCE'
INSTANCE_ID = 'INSTANCE_ID'
INTERVAL_STATE = 'INTERVAL_STATE'
INTERSECTS = 'INTERSECTS'
INSTANCE_ID = 'INSTANCE_ID'
IO_FORMATTER = 'IO_FORMATTER'
IREGULAR = 'IREGULAR'
IS_ALERT = 'IS_ALERT'
IS_THREAD = 'IS_THREAD'
IS_NEW_ALERT = 'IS_NEW_ALERT'
ITERATION = 'ITERATION'
IS_ANOM_PRESENCE = 'IS_ANOM_PRESENCE'
IS_ANOM_ENVIRON = 'IS_ANOM_ENVIRON'
TAGS = 'TAGS'

JETSON_VIDEO_STREAM = 'JETSON_VIDEO_STREAM'
LAST_COUNT = 'LAST_COUNT'
LAST_EXPORTED_STATE = 'LAST_EXPORTED_STATE'
LAST_EXPORTED_DATE = 'LAST_EXPORTED_DATE'
LAST_EXPORT = 'LAST_EXPORT'
LAST_READ = 'LAST_READ'
LAST_SEEN_TIME = 'LAST_SEEN_TIME'
LAST_TEMPERATURE = 'LAST_TEMPERATURE'
LAST_HUMIDITY = 'LAST_HUMIDITY'
LAST_STRUCT_TIME = 'LAST_STRUCT_TIME'
LAST_PAYLOAD = 'LAST_PAYLOAD'
LINE = 'LINE'
LEFT = 'LEFT'
LICENSE_PLATE_TRACKER = 'LICENSE_PLATE_TRACKER'
LIVE_FEED = 'LIVE_FEED'
LOCATION = 'LOCATION'
LOCATION_NAME = 'LOCATION_NAME'
LOCATIONS = 'LOCATIONS'
LOCK_CMD = 'lock_cmd'
MAIN = 'MAIN'
MASTER = 'MASTER'
MACHINE_IP = 'MACHINE_IP'
MACHINE_MEMORY = 'MACHINE_MEMORY'
MAX_IDLE_TIME = 'MAX_IDLE_TIME'
MESSAGE = 'MESSAGE'
MESSAGE_ID = 'MESSAGE_ID'
MESSAGE_INFO = 'MESSAGE_INFO'
MESSAGES = 'MESSAGES'
METADATA = 'METADATA'
MEMORY_FRACTION = 'MEMORY_FRACTION'
MIN_PIXELS_FROM_EDGE = 'MIN_PIXELS_FROM_EDGE'
MIN_PERS_QUEUE = 'MIN_PERS_QUEUE'
MIN_PERS_QUEUE_SIZE = 'MIN_PERS_QUEUE_SIZE'
MIN_LENGTH = 'MIN_LENGTH'
MIN_TRAIN_STEPS = 'MIN_TRAIN_STEPS'

MODEL_SERVING_PROCESS = 'MODEL_SERVING_PROCESS'
MODEL_CLASSES_FILENAME = 'MODEL_CLASSES_FILENAME'
MODEL_WEIGHTS_FILENAME = 'MODEL_WEIGHTS_FILENAME'
SECOND_STAGE_MODEL_WEIGHTS_FILENAME = 'SECOND_STAGE_MODEL_WEIGHTS_FILENAME'
SECOND_STAGE_MODEL_WEIGHTS_FILENAME = 'SECOND_STAGE_MODEL_WEIGHTS_FILENAME'
CUDNN_BENCHMARK = 'CUDNN_BENCHMARK'

META_TYPE = 'META_TYPE'
MODULE = 'MODULE'
NAME = 'NAME'
NODE = 'NODE'
NO_WITNESS = 'NO_WITNESS'
NON_VIDEO_PLUGIN = 'NON_VIDEO_PLUGIN'
NR_FRAME = 'NR_FRAME'
NR_INF = 'NR_INF'
NR_INFERENCES = 'NR_INFERENCES'
NR_OBJECTS = 'NR_OBJECTS'
NR_PAYLOADS = 'NR_PAYLOADS'
NR_PERSONS = 'NR_PERSONS'
NR_PERS = 'NR_PERS'
NR_PERS_FLOOR = 'NR_PERS_FLOOR'
NR_SKIP_FRAMES = 'NR_SKIP_FRAMES'
NR_STREAMS = 'NR_STREAMS'
NR_STREAMS_DATA = 'NR_STREAMS_DATA'
OBJECT_SUBPART = 'OBJECT_SUBPART'
OBJECT_TYPE = 'OBJECT_TYPE'
OBJECTS = 'OBJECTS'
OBSTRUCTION = 'OBSTRUCTION'
OBSTRUCTION_FINGER = 'OBSTRUCTION_FINGER'
OBSTRUCTION_OTHER = 'OBSTRUCTION_OTHER'
OBSTRUCTION_RESIZE = 'OBSTRUCTION_RESIZE'
OFF = 'OFF'
ON = 'ON'
OPEN = 'OPEN'
OUTPUT = 'OUTPUT'
OUTPUT_FPS = 'OUTPUT_FPS'
OUTPUT_PATH = 'OUTPUT_PATH'
OUTPUT_PATH_PREVIEW = 'OUTPUT_PATH_PREVIEW'
OUTPUT_SCALE_FACTOR = 'OUTPUT_SCALE_FACTOR'
OUTPUT_VIDEO = 'OUTPUT_VIDEO'
OUTPUT_VIDEO_PREVIEW = 'OUTPUT_VIDEO_PREVIEW'
ORIGINAL_FRAME = 'ORIGINAL_FRAME'
PLUGINS_ON_THREADS = 'PLUGINS_ON_THREADS'
PRC_PERSON_BLUR = 'PRC_PERSON_BLUR'
PRC_VEHICLES_BLUR = 'PRC_VEHICLES_BLUR'
PERSONS_CLASSES = 'PERSONS_CLASSES'
PLUGIN_RESULTS = 'PLUGIN_RESULTS'
POINTS = 'POINTS'
PLUGINS_RESULTS = 'PLUGINS_RESULTS'
PLUGINS = 'PLUGINS'
PREDICTION_STEPS = 'PREDICTION_STEPS'
PROCESSED_COUNT = 'PROCESSED_COUNT'
PROCESSED_DATE = 'PROCESSED_DATE'
PROCESSING_RESULTS_CSV = 'PROCESSING_RESULTS_CSV'
PRC_INTERSECT = 'PRC_INTERSECT'
PRC_GOLD = 'PRC_GOLD'
PRIORITY = 'PRIORITY'
PROB_PRC = 'PROB_PRC'
PROGRESS = 'PROGRESS'
PROCESS_MEMORY = 'PROCESS_MEMORY'
PROCESS_DELAY = 'PROCESS_DELAY'
PREDICTION_STEPS = 'PREDICTION_STEPS'
RESULT = 'RESULT'
QUALITY = 'QUALITY'
QUEUE_DIRECTION = 'QUEUE_DIRECTION'
QUERY = 'QUERY'
RATIO = 'RATIO'
RECONNECTABLE = 'RECONNECTABLE'
RECORD = 'RECORD'
REGULAR = 'REGULAR'
REID = 'REID'
REIDENTIFICATIONS = 'REIDENTIFICATIONS'
REIDENTIFICATED = 'REIDENTIFICATED'
REIDENTIFICATED_TIME = 'REIDENTIFICATED_TIME'
RESULTS = 'RESULTS'
RENAME_OBJECT_TYPE = 'RENAME_OBJECT_TYPE'
RIGHT = 'RIGHT'
SAFETY_HELMET = 'SAFETY_HELMET'
SAVE_BUFFER = 'SAVE_BUFFER'
SAVE_INTERVAL = 'SAVE_INTERVAL'
SAVE_SECONDS = 'SAVE_SECONDS'
SECONDS_HEARTBEAT = 'SECONDS_HEARTBEAT'
SECONDS_READ = 'SECONDS_READ'
SECONDS_DISMISS_ALERT = 'SECONDS_DISMISS_ALERT'
SECONDS_RAISE_ALERT = 'SECONDS_RAISE_ALERT'
SECS_READ = 'SECS_READ'
SECS_RAISE_ALERT = 'SECS_RAISE_ALERT'
SECS_DISMISS_ALERT = 'SECS_DISMISS_ALERT'
SECS_NEXT_DECISION = 'SECS_NEXT_DECISION'
SELECTIVE_ZONES = 'SELECTIVE_ZONES'
SEND_EMAIL = '_H_SEND_EMAIL'
SEND_PRIORITY = 'SEND_PRIORITY'
SEND_PRIORITY_HIGH = 'SEND_PRIORITY_HIGH'
SEND_PRIORITY_MEDIUM = 'SEND_PRIORITY_MEDIUM'
SEND_PRIORITY_NORMAL = 'SEND_PRIORITY_NORMAL'
SERVER_ID = 'SERVER_ID'
SERVING_PIDS = 'SERVING_PIDS'
SHARED = 'SHARED'
SHM_OUTPUT = 'SHM_OUTPUT'
SIGNATURE = 'SIGNATURE'
SIMPLE_WITNESS = 'SIMPLE_WITNESS'
SLOW = 'SLOW'
SMALL_CIRCLE_RADIUS_PERCENT = 'SMALL_CIRCLE_RADIUS_PERCENT'
STAGE_DOWNLOAD = 'DOWNLOAD'
STAGE_PROCESS = 'PROCESS'
STAGE_PING = 'PING'
STAGE_SEARCH_WORKERS = 'SEARCH_WORKERS'
STANDING_STILL = 'STANDING_STILL'
STANDING_STILL_TIME = 'STANDING_STILL_TIME'
START = 'START'
START_TIME = 'START_TIME'
STATE = 'STATE'
STATUS = 'STATUS'
STATUS_CHANGED = 'STATUS_CHANGED'
STEP = 'STEP'
STEP_MINUTES = 'STEP_MINUTES'
STILL_SECONDS = 'STILL_SECONDS'
STILL_TIME = 'STILL_TIME'
STREAM = 'STREAM'
STREAM_ID = 'STREAM_ID'
STREAM_CONFIG_METADATA = 'STREAM_CONFIG_METADATA'
STREAM_DEBUG_DATA = 'STREAM_DEBUG_DATA'
STREAM_IMG = 'STREAM_IMG'
STREAM_INFER = 'STREAM_INFER'
STREAM_INFER_INTERSECT = 'STREAM_INFER_INTERSECT'
STREAM_MESSAGES = 'STREAM_MESSAGES'
STREAM_PLUGINS_RESULTS = 'STREAM_PLUGINS_RESULTS'
STREAM_STEP_RESULTS = 'STREAM_STEP_RESULTS'
STREAMS = 'STREAMS'
STREAMS_KEY = 'STREAMS_KEY'
STRIDE = 'STRIDE'
SYSTEM_TIME = 'SYSTEM_TIME'
RESERVED_MEM = 'RESERVED_MEM'
TARGET_SUBPART = 'TARGET_SUBPART'
THR_MASK = 'THR_MASK'
THREADS_PREFIX = 'S_'
THRESHOLD = 'THRESHOLD'
TIME = 'TIME'
TOPIC_DEVICE_SPECIFIC = 'TOPIC_DEVICE_SPECIFIC'
TOPIC_SEND = 'TOPIC_SEND'
TOPIC_RECV = 'TOPIC_RECV'
TRACKING_ALIVE_TIME = 'TRACKING_ALIVE_TIME'
TRACKING_MODE = 'TRACKING_MODE'
TRACKING_OBJECTS = 'TRACKING_OBJECTS'
PAYLOADS_CHANNEL = 'PAYLOADS_CHANNEL'
CONFIG_CHANNEL = 'CONFIG_CHANNEL'
TOPIC_SEND_DEVICE_SPECIFIC = 'TOPIC_SEND_DEVICE_SPECIFIC'
TOPIC_RECV_DEVICE_SPECIFIC = 'TOPIC_RECV_DEVICE_SPECIFIC'
TOTAL_MEM = 'TOTAL_MEM'
TOTAL_DISK = 'TOTAL_DISK'
TOTAL_NR_PERS = 'TOTAL_NR_PERS'
TOTAL_MESSAGES = 'TOTAL_MESSAGES'
TIME_IN_TARGET = 'TIME_IN_TARGET'
TRACK_ID = 'TRACK_ID'
TRACKING_STARTED_AT = 'TRACKING_STARTED_AT'
TRAIN_PERIODS = 'TRAIN_PERIODS'
TRAIN_HIST = 'TRAIN_HIST'
TRAIN_STEPS = 'TRAIN_STEPS'
TRANSCODER_H = 'TRANSCODER_H'
TRANSCODER_W = 'TRANSCODER_W'
TURNSTILE_SECONDS_INTERVAL = 'TURNSTILE_SECONDS_INTERVAL'
QOS = 'QOS'
UNKNOWN = 'UNKNOWN'
UPLOAD_MOVIE = 'UPLOAD_MOVIE'
UPLOAD_ZIP = 'UPLOAD_ZIP'
UPTIME = 'UPTIME'
URL = 'URL'
URL_CONFIG = 'URL_CONFIG'
URL_DOWNLOAD = 'URL_DOWNLOAD'
URL_DETECTIONS = 'URL_DETECTIONS'
URL_MODELS_ZOO = 'URL_MODELS_ZOO'
URLS = 'URLS'
USE = 'USE'
VALUE = 'VALUE'
VANISHING = 'VANISHING'
VEHICLE_CLASSES = 'VEHICLE_CLASSES'
VIDEO_FILE = 'VIDEO_FILE'
VIDEO_STREAM = 'VIDEO_STREAM'
VERSION = 'VERSION'
WEAPON_ASSAILANT = 'WEAPON_ASSAILANT'
WIDTH_PRC = 'WIDTH_PRC'
WHITELIST = 'WHITELIST'
WHITELISTED = 'WHITELISTED'
WHITELISTED_TIME = 'WHITELISTED_TIME'
TIMERS = 'TIMERS'
TYPE = 'TYPE'
THR_ANGLE_ALERT = 'THR_ANGLE_ALERT'
TIMESTAMP = 'TIMESTAMP'
TLBR_POS = 'TLBR_POS'
TLBR_POS_TRACK = 'TLBR_POS_TRACK'
TOP = 'TOP'
TOP_N_SOFTMAX = 'TOP_N_SOFTMAX'
TURNSTILE_SEC = 'TURNSTILE_SEC'
ZONE = 'ZONE'
ZONES = 'ZONES'
ZONE_NR_PERS = 'ZONE_NR_PERS'
ZIP_SIZE = 'ZIP_SIZE'
WRONG = 'WRONG'
WARMUP_FRAMES = 'WARMUP_FRAMES'
WARMUP_ITERATIONS = 'WARMUP_ITERATIONS'


NO_MESSAGE = "STATUS UNCHANGED"
INFO_MSG = "INFO_MSG"

# LISTS
LIST_MOVIE = ['.avi', '.mp4', '.mkv']
LIST_IMAGE = ['.png', '.jpg', '.jpeg', '.bmp']
LIST_PKL = ['.pkl', '.pk']


PLUGIN_INSTANCE_PARAMETER_LIST = 'PLUGIN_INSTANCE_PARAMETER_LIST'


# DRAWING
# BGRs
DARK_BLUE = (109, 45, 11)
BLUE = (255, 0, 0)
LIGHT_BLUE = (230, 200, 100)
ORANGE = (19, 149, 250)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (98, 222, 250)
RED = (100, 70, 248)
GREEN = (20, 180, 20)
DARK_GREEN = (20, 120, 20)
PALE_GREEN = (60, 230, 170)
DEEP_RED = (20, 20, 170)
BROWN = (42, 42, 165)
PURPLE = (211, 0, 148)


# OTHER
LIGHT_BLUR = (254, 255, 129)


LOW_DRAWING = 'LOW_DRAWING'

# END DRAWING


# DetectionClasses
CAR = 'car'
FACE = 'face'
PERSON = 'person'
PERSOANA = 'persoana'


BLUR_PERSON_DEFAULT = 'default'
BLUR_PERSON_ADAPTIVE = 'adaptive'

RECTANGLE_TYPE_DEFAULT = 'default'
RECTANGLE_TYPE_CORNERS = 'corners'

# #CAMERA QUALITY CHECK
# CQC_COMPRESS = 'COMPRESS'
# CQC_DIRT = 'DIRT'
# CQC_EMP = 'EMP'
# CQC_FOG = 'FOG'
# CQC_GOOD = 'GOOD'
# CQC_ICE = 'ICE'
# CQC_OBSCURED = 'OBSCURED'
# CQC_SNOW = 'SNOW'
# CQC_WATER = 'WATER'

# GRID HEATMAP
GRID_X_COUNT = 'GRID_X_COUNT'
GRID_Y_COUNT = 'GRID_Y_COUNT'
REF_AREA_TL = 'REF_AREA_TL'
REF_AREA_TR = 'REF_AREA_TR'
REF_AREA_BL = 'REF_AREA_BL'
REF_AREA_BR = 'REF_AREA_BR'
TRANSFORM_WHOLE = 'TRANSFORM_WHOLE'
HEAT_RANGE = 'HEAT_RANGE'
HEATMAP_RECORD_INTERVAL = 'HEATMAP_RECORD_INTERVAL'
HEATMAP_PERSON_LOW = 'HEATMAP_PERSON_LOW'
HEATMAP_PERSON_LOW_DEFAULT = 0.1

# LOCATIONS
SENSORS_LOCATION = 'naeural_core.utils.sensors'

CAP_ACTUAL_DPS = 'CAP_ACTUAL_DPS'
CAP_CACHE_MAX = 'CAP_CACHE_MAX'
CAP_CACHE_LEN = 'CAP_CACHE_LEN'
CAP_LIVE = 'CAP_LIVE'
CAP_RECONN = 'CAP_RECONNECTABLE'


# DATA ACQUISITION, PAYLOADS & COMM RESOLUTIONS & other stuff
CAPTURE_MANAGER = 'CAPM'

NR_CAPTURES = 'NR_CAPTURES'

TRANSCODER_FAILS = 200
MAIN_LOOP_RESOLUTION = 20

DEFAULT_SECONDS_HEARTBEAT = 10
MAX_SECONDS_HEARTBEAT = 10

CAP_RESOLUTION_DEFAULT = 4
# we have queue protection so we do not need large queue
CAP_RESOLUTION_NON_LIVE_DEFAULT = 32

INIT_DATA = 'INIT_DATA'
CAP_RESOLUTION = 'CAP_RESOLUTION'

CAPTURES_PER_SEC = 'CAPTURES_PER_SEC'
STREAM_TIME = 'STREAM_TIME'

CAPTURE_STATS_DISPLAY = 'CAPTURE_STATS_DISPLAY'
CAPTURE_STATS_DISPLAY_DEFAULT = 120  # in seconds

OBSOLETE_FLAGS = ['CAP_READ_DELAY',]

PLUGIN_EXCEPTION_DELAY = 10


COMM_SEND_BUFFER = 1000
COMM_RECV_BUFFER = 100
COMM_SECS_SHOW_INFO = 60


# Payload constants
IDLE_PLUGIN_ITERS = CAP_RESOLUTION_DEFAULT * 20
PLUGIN_REAL_RESOLUTION = 'PLUGIN_REAL_RESOLUTION'
PLUGIN_LOOP_RESOLUTION = 'PLUGIN_LOOP_RESOLUTION'


RESIZE_H = 'RESIZE_H'
RESIZE_W = 'RESIZE_W'
RESIZE_WITNESS = 'RESIZE_WITNESS'
FRAME_CROP = 'FRAME_CROP'
CAPTURE_CROP = 'CAPTURE_CROP'  # alias for FRAME_CROP

FORCED_DELAY = 5

# END DATA AREA


# INFERENCE ENGINE

PICKED_INPUT = 'PICKED_INPUT'
RUNS_ON_EMPTY_INPUT = 'RUNS_ON_EMPTY_INPUT'
MODEL_INSTANCE_ID = 'MODEL_INSTANCE_ID'


TIMER_FILTER_RESULTS = 'filter_results'
TIMER_LOAD_GRAPH = 'load_graph'
TIMER_PREDICT = 'pred'
TIMER_PREDICT_BATCH = 'pred_batch'
TIMER_PREPARE_PAYLOAD = 'prep_payload'
TIMER_PREPARE_RESULTS = 'prep_results'
TIMER_PRE_PROCESS = 'pre_proc'
TIMER_POST_PROCESS = 'post_proc'
TIMER_PREPROCESS_INPUTS = 'pre_proc_inp'
TIMER_POST_PROCESS_INFERENCE = 'post_proc_inf'
TIMER_POST_PROCESS_OUTPUTS = 'post_proc_out'
TIMER_RUN_INFERENCE = 'run_inf'
TIMER_PACK_RESULTS = 'pack_res'
TIMER_SESSION_RUN = 'sess_run'

BATCH_STRATEGY_DEFAULT = 'DEFAULT'
BATCH_STRATEGY_MOST_COMMON_SHAPE = 'MOST_COMMON_SHAPE'
BATCH_STRATEGY_PER_SHAPE = 'PER_SHAPE'
CLASSES = 'CLASSES'
CONFIG = 'CONFIG'
CONFIG_INFERENCE = 'CONFIG_INFERENCE'
CONFIG_PLUGINS = 'CONFIG_PLUGINS'
CONFIG_STREAMS = 'CONFIG_STREAMS'
ERROR = 'ERROR'
GRAPH = 'GRAPH'
IMG = 'IMG'
IMG_H = 'IMG_H'
IMG_W = 'IMG_W'
IOU = 'IOU'
INPUT_TENSORS = 'INPUT_TENSORS'
INFERENCES = 'INFERENCES'
IOU_THRESHOLD = 'IOU_THRESHOLD'
MODEL_THRESHOLD = 'MODEL_THRESHOLD'
METADATA = 'METADATA'
NMS = 'NMS'
OUTPUT_TENSORS = 'OUTPUT_TENSORS'
STYLE_IMAGE = 'STYLE_IMAGE'
SYSTEM_TIME = 'SYSTEM_TIME'
VER = 'VER'


# default confidence threshold should be high in most cases OR
# we can use AlertHelper with percentages and mean of 0.51 or higher


# TORCH constants
WARMUP_BATCH = 'WARMUP_BATCH'
USE_AMP = 'USE_AMP'
URL_CLASS_NAMES = 'URL_CLASS_NAMES'
MODEL_NAME = 'MODEL_NAME'
NMS_CONF_THR = 'NMS_CONF_THR'
NMS_IOU_THR = 'NMS_IOU_THR'
NMS_MAX_DET = 'NMS_MAX_DET'
YAML_PATH = 'YAML_PATH'
DEFAULT_BATCH = 'DEFAULT_BATCH'
URL_BY_BATCH = 'URL_BY_BATCH'
DEFAULT_DEVICE = 'DEFAULT_DEVICE'
USE_FP16 = 'USE_FP16'
GPU_PREPROCESS = 'GPU_PREPROCESS'
MAX_BATCH_FIRST_STAGE = 'MAX_BATCH_FIRST_STAGE'
MAX_BATCH_SECOND_STAGE = 'MAX_BATCH_SECOND_STAGE'


# INFERENCE
TIMER_CALCULATE_BENCHMARK_TIME = 'calculate_benchmark_time'
TIMER_INFER = 'timer_infer'
TIMER_LOAD_GRAPH = 'load_graph'
TIMER_POSTPROCESS_BOXES = 'postprocess_boxes'
TIMER_POSTPROCESS_IMAGES = 'postprocess_images'
TIMER_PREDICT = 'predict'
TIMER_PREDICT_BATCH = 'predict_batch'
TIMER_PREPARE_PAYLOAD = 'prepare_payload'
TIMER_PREPARE_RESULTS = 'prepare_results'
TIMER_PREPROCESS_IMAGES = 'preprocess_images'
TIMER_RUN_INFERENCE = 'run_inference'
TIMER_SESSION_RUN = 'session_run'

BATCH_STRATEGY_DEFAULT = 'DEFAULT'
BATCH_STRATEGY_MOST_COMMON_SHAPE = 'MOST_COMMON_SHAPE'
BATCH_STRATEGY_PER_SHAPE = 'PER_SHAPE'

CONFIG = 'CONFIG'
CONFIG_INFERENCE = 'CONFIG_INFERENCE'
CONFIG_PLUGINS = 'CONFIG_PLUGINS'
CONFIG_STREAMS = 'CONFIG_STREAMS'

GRAPH = 'GRAPH'
GRAPH_TYPE_PB = 'pb'
GRAPH_TYPE_ONNX = 'onnx'
GRAPH_TYPE_TENSORFLOW_TRT = 'tensorflow_trt'
GRAPH_TYPE_PYTORCH_TRT = 'pytorch_trt'
GRAPH_TYPE_KERAS = 'keras'
GRAPH_TYPE_PYTORCH = 'pytorch'

GRAPH_TASK_CLASSIFICATION = 'classification'
GRAPH_TASK_DETECTION = 'detection'

FRAMEWORK_TENSORFLOW = 'tensorflow'
FRAMEWORK_PYTORCH = 'pytorch'

INFERENCE_GRAPHS = 'INFERENCE_GRAPHS'
INFERENCE_META_GRAPHS = 'INFERENCE_META_GRAPHS'
INFERENCE_ON_FULL_IMAGE = 'INFERENCE_ON_FULL_IMAGE'
IOU_THRESHOLD = 'IOU_THRESHOLD'

LINKED_INSTANCES = 'LINKED_INSTANCES'
LINKED_SERVER = 'LINKED_SERVER'


LOCATIONS = 'LOCATIONS'

MODEL_THRESHOLD = 'MODEL_THRESHOLD'

PLUGINS = 'PLUGINS'
SIGNATURE = 'SIGNATURE'
USE_GRAPHS = 'USE_GRAPHS'


ALIMENTARE_NEPERMISA = 'ALIMENTARE_NEPERMISA'
CAMERA_QUALITY_CHECK = 'CAMERA_QUALITY_CHECK'
COVID_MASK = 'COVID_MASK'
EFF_DET0 = 'EFF_DET0'
FIRE_SMOKE = 'FIRE_SMOKE'
FACE_DETECTION = 'FACE_DETECTION'
GASTRO = 'GASTRO'
LP_DETECTION = 'LP_DETECTION'
LPDV2 = 'LPDV2'
LPR = 'LPR'
MERCHANDISER = 'MERCHANDISER'
OMV_EMPLOYEE = 'OMV_EMPLOYEE'
TF_YOLO = 'TF_YOLO'
TF_ODAPI1 = 'TF_ODAPI1'
TF_ODAPI1_OIDV4 = 'TF_ODAPI1_OIDV4'
TF_ODAPI1_TRAFFICSIGNS = 'TF_ODAPI1_TRAFFICSIGNS'
TF_ODAPI2 = 'TF_ODAPI2'

LST_VEHICLES = ['car', 'truck', 'bus']

WEEKDAYS = [
  'MONDAY',
  'TUESDAY',
  'WEDNESDAY',
  'THURSDAY',
  'FRIDAY',
  'SATURDAY',
  'SUNDAY'
]


class ThAnchor:
  ANCHOR_UPDATED_AT = "ANCHOR_UPDATED_AT"
  ANCHOR_METADATA = "ANCHOR_METADATA"
  ANCHOR_IMAGES = "ANCHOR_IMAGES"
  ANCHOR_RELOAD_TIME = "ANCHOR_RELOAD_TIME"

  PREDICTIONS = "PREDICTIONS"


class DatasetBuilder:
  FILE = 'FILE'
  LABEL_TEMPLATE_TYPE = 'LABEL_TEMPLATE_TYPE'
  LABEL_FILE_TEMPLATE = 'LABEL_FILE_TEMPLATE'
  LABEL_EXTENSION = 'LABEL_EXTENSION'
  X_EXTENSION = 'X_EXTENSION'
  DATASET_BUILDER = 'DATASET_BUILDER'
  DATASET_NAME = 'DATASET_NAME'
  VALID_FILE_TYPES = ['FILE', None]
  INFERENCE_MAPPING = 'INFERENCE_MAPPING'


class ForceStopException(Exception):
  pass

CONFIG_STARTUP_MANDATORY_KEYS = [
  EE_ID, 
  CONFIG_STARTUP_v2.SECURED,
  "MAIN_LOOP_RESOLUTION",
  SECONDS_HEARTBEAT,
  CONFIG_STARTUP_v2.K_CONFIG_RETRIEVE,
  PLUGINS_ON_THREADS,
]

try:
  from constants import *
except:
  pass