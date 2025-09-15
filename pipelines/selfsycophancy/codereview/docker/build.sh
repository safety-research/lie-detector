#!/bin/bash
# Docker build script with intelligent caching

set -euo pipefail

# Configuration
IMAGE_NAME="sycophancy-experiment"
IMAGE_TAG="latest"
DOCKERFILE_PATH="docker/Dockerfile"
CONTEXT_PATH="docker"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if image exists and get its age
check_image_exists() {
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}:${IMAGE_TAG}$"
}

# Get image creation time in seconds since epoch
get_image_age() {
    if check_image_exists; then
        docker inspect -f '{{.Created}}' "${IMAGE_NAME}:${IMAGE_TAG}" | xargs -I {} date -d {} +%s
    else
        echo "0"
    fi
}

# Check if Dockerfile has been modified
dockerfile_modified_since() {
    local image_time=$1
    local dockerfile_time=$(stat -c %Y "$DOCKERFILE_PATH" 2>/dev/null || stat -f %m "$DOCKERFILE_PATH" 2>/dev/null)
    [ "$dockerfile_time" -gt "$image_time" ]
}

# Main build logic
main() {
    echo -e "${GREEN}Docker Build Script for Self-Sycophancy Experiment${NC}"
    echo "=================================================="
    
    # Check if force rebuild is requested
    if [ "${1:-}" == "--force" ]; then
        echo -e "${YELLOW}Force rebuild requested${NC}"
        BUILD_REQUIRED=true
    else
        BUILD_REQUIRED=false
        
        # Check if image exists
        if ! check_image_exists; then
            echo -e "${YELLOW}Image ${IMAGE_NAME}:${IMAGE_TAG} not found. Build required.${NC}"
            BUILD_REQUIRED=true
        else
            # Check if Dockerfile has been modified
            IMAGE_TIME=$(get_image_age)
            if dockerfile_modified_since "$IMAGE_TIME"; then
                echo -e "${YELLOW}Dockerfile has been modified. Rebuild required.${NC}"
                BUILD_REQUIRED=true
            else
                echo -e "${GREEN}Image is up to date. No rebuild needed.${NC}"
                
                # Show image info
                echo -e "\nImage details:"
                docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}"
            fi
        fi
    fi
    
    # Build if required
    if [ "$BUILD_REQUIRED" = true ]; then
        echo -e "\n${GREEN}Building Docker image...${NC}"
        
        # Enable BuildKit
        export DOCKER_BUILDKIT=1
        
        # Build with cache
        docker build \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            --cache-from "${IMAGE_NAME}:${IMAGE_TAG}" \
            -t "${IMAGE_NAME}:${IMAGE_TAG}" \
            -f "$DOCKERFILE_PATH" \
            "$CONTEXT_PATH"
        
        if [ $? -eq 0 ]; then
            echo -e "\n${GREEN}Build completed successfully!${NC}"
            
            # Show new image info
            echo -e "\nNew image details:"
            docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}"
            
            # Prune dangling images
            echo -e "\n${GREEN}Cleaning up dangling images...${NC}"
            docker image prune -f
        else
            echo -e "\n${RED}Build failed!${NC}"
            exit 1
        fi
    fi
    
    # Show cache usage
    echo -e "\n${GREEN}Docker cache usage:${NC}"
    docker system df --format "table {{.Type}}\t{{.Total}}\t{{.Active}}\t{{.Size}}\t{{.Reclaimable}}"
}

# Run main function
main "$@"