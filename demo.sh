# Configurable parameters (can be modified as needed)
API_PORT=8002                # API service port
API_ENDPOINT="localhost"     # API service address
MODEL_NAME="qwen2_vl"        # Model name
SERVER_PORT=7880             # Gradio service port
SERVER_NAME="127.0.0.1"      # Gradio service address
CUDA_DEVICE="0"              # CUDA device ID
SHARE=false                  # Whether to enable public sharing

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-port)
      API_PORT="$2"
      shift 2
      ;;
    --api-endpoint)
      API_ENDPOINT="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --server-port)
      SERVER_PORT="$2"
      shift 2
      ;;
    --server-name)
      SERVER_NAME="$2"
      shift 2
      ;;
    --cuda-device)
      CUDA_DEVICE="$2"
      shift 2
      ;;
    --share)
      SHARE=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Build Share parameters
SHARE_ARGS=""
if [ "$SHARE" = true ]; then
  SHARE_ARGS="--share"
fi

# Start API service (run in background) and save log
echo "Starting API service and loading model..."
API_PORT=$API_PORT CUDA_VISIBLE_DEVICES=$CUDA_DEVICE llamafactory-cli api src/inference/config/qwen2_vl.yaml > model_loading.log 2>&1 &
API_PID=$!

# Check if model directory exists
if [ ! -d "checkpoints/pretrained/JarvisArt-preview" ]; then
    echo "❌ Error: Model directory does not exist. Please ensure you've downloaded model weights to checkpoints/pretrained/JarvisArt-preview directory"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Check if model loading was successful
sleep 3
if ! ps -p $API_PID > /dev/null; then
    echo "❌ Error: API service failed to start. Model may not be loading properly. See log for details:"
    cat model_loading.log
    exit 1
fi

# Cleanup function
cleanup() {
    echo "Stopping background API service..."
    kill $API_PID 2>/dev/null
    wait $API_PID 2>/dev/null
    rm -f model_loading.log
    exit
}

# Wait for model to load and check status
echo "Waiting for model to load..."
sleep 5
if grep -q "error\|Error\|ERROR\|failed\|Failed" model_loading.log; then
    echo "❌ Error loading model. Details:"
    cat model_loading.log
    cleanup
    exit 1
else
    echo "✅ Model loaded successfully!"
fi

# API configuration parameters
api_key="0"

# Construct JSON data
json_data=$(cat <<EOF
{
  "model": "gpt",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "hello!"
        }
      ]
    }
  ]
}
EOF
)

# Retry loop
echo "Testing API connection..."
max_retries=20
retry_count=0

while [ $retry_count -lt $max_retries ]; do
  # Send POST request
  response=$(curl -s -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $api_key" \
    -d "$json_data" \
    --insecure \
    "http://${API_ENDPOINT}:${API_PORT}/v1/chat/completions" 2>/dev/null)
  
  # Extract status code and response body
  http_code=${response: -3}
  response_body=${response%???}

  # Check status code
  if [ "$http_code" -eq 200 ]; then
    echo "✅ API request successful!"
    echo "Response:"
    echo "$response_body" | jq .  # Format output using jq
    break
  else
    retry_count=$((retry_count + 1))
    echo "⏳ API initialization in progress... Status code: $http_code"
    echo "Error response: $response_body"
    
    if [ $retry_count -lt $max_retries ]; then
      echo "Waiting for model to be fully loaded... (Attempt $retry_count of $max_retries)"
      sleep 3
    else
      echo "Maximum retries reached. API may not be functioning correctly."
      echo "⚠️ Continuing anyway, but the application might not work properly."
      echo "Please check if the model failed to load or if there are other issues."
    fi
  fi
done

trap cleanup SIGINT

# Run Python program, passing all relevant parameters
echo "Starting JarvisArt demo interface..."
python demo.py \
  --api_endpoint $API_ENDPOINT \
  --api_port $API_PORT \
  --model_name $MODEL_NAME \
  --server_port $SERVER_PORT \
  --server_name $SERVER_NAME \
  $SHARE_ARGS
