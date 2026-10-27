#import <Metal/Metal.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

// C++ interface to load the embedded metallib without exposing ObjC types
extern "C" {
  void* loadEmbeddedMetalLibrary(void* device, const char** errorMsg) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    NSError* error = nil;

    id<MTLLibrary> library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(mtlDevice, &error);

    if (!library && errorMsg && error) {
      *errorMsg = strdup([error.localizedDescription UTF8String]);
    }

    // Manually retain since we're not using ARC
    // The caller will wrap in NS::TransferPtr which assumes ownership
    if (library) {
      [library retain];
    }
    return (__bridge void*)library;
  }

  // Get PyTorch's MPS device (returns id<MTLDevice> as void*)
  void* getMPSDevice() {
    return (__bridge void*)at::mps::MPSDevice::getInstance()->device();
  }
}

// Encode work on PyTorch's current MPS stream.
//
// `encode` runs on the stream's serial dispatch queue and receives the
// stream's compute command encoder (id<MTLComputeCommandEncoder> as void*).
// The encoder is owned by the stream (kernel coalescing): `encode` must not
// call endEncoding() or commit, and must not throw. The work is ordered with
// all other PyTorch MPS ops on the stream and is committed by PyTorch at the
// next synchronization point.
//
// C++ linkage so `encode` matches an ordinary C++ function pointer. The stream
// is read on the caller thread so this follows that thread's current MPS
// stream (including torch.mps.stream on recent torch).
void mpsEncodeOnStream(void (*encode)(void* encoder, void* ctx), void* ctx) {
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^() {
    @autoreleasepool {
      encode((__bridge void*)stream->commandEncoder(), ctx);
    }
  });
}
