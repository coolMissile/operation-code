# Use alpine as the base image for the build stage
FROM alpine:latest AS builder

# Install musl-dev, gcc, and make in the build stage
RUN apk add --no-cache build-base

# Set the working directory
WORKDIR /app

# Copy the project files into the Docker image
COPY . .

RUN mkdir -p SRC/OBJ

RUN make CFLAGS="-O3 -Wall -IINCLUDE -DTWO_LEVEL_TREE -g -flto -fcommon -static"

# Create the final stage
FROM scratch

# Copy the built static binary from the builder stage
COPY --from=builder /app/LKH /bin/LKH

# Set the working directory
WORKDIR /app

# Set the entrypoint to run the built executable
ENTRYPOINT ["/bin/LKH"]
