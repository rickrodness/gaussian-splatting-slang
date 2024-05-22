#include <slang.h>

// Structure to hold Gaussian parameters
struct Gaussian {
    float2 xy;       // Center coordinates of the Gaussian
    float3 conic;    // Conic coefficients
    float3 rgb;      // RGB color values
    float opacity;   // Opacity value
};

// Structure to hold the parameters for the backward pass
struct RasterizeBackwardParams {
    float2 tileBounds;   // Tile Bounds
    float2 imgSize;      // Image Size
    int channels;        // Number of Channels 
    StructuredBuffer<int> gaussiansIdsSorted; // Sorted Gaussian IDs
    StructuredBuffer<int2> tileBins;          // Tile bins
    StructuredBuffer<float2> xys;             // XY coordinates
    StructuredBuffer<float3> conics;          // Conic coefficients
    StructuredBuffer<float> rgbs;             // RGB values
    StructuredBuffer<float> opacities;        // Opacity values
    StructuredBuffer<float> background;       // Background values
    StructuredBuffer<float> finalTs;          // Final T values
    StructuredBuffer<int> finalIndex;         // Final index values
    StructuredBuffer<float> vOutput;          // Output values
    StructuredBuffer<float> vOutputAlpha;     // Output alpha values
    RWStructuredBuffer<float2> vXY;           // Gradient with respect to XY
    RWStructuredBuffer<float2> vXYAbs;        // Absolute gradient with respect to XY
    RWStructuredBuffer<float3> vConic;        // Gradient with respect to conic coefficients
    RWStructuredBuffer<float> vRgb;           // Gradient with respect to RGB values
    RWStructuredBuffer<float> vOpacity;       // Gradient with respect to opacity values
};

// Function to perform warp-level sum reduction for a single float value
float warpSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Function to perform warp-level sum reduction for a float2 value
float2 warpSum2(float2 val) {
    val.x = warpSum(val.x);
    val.y = warpSum(val.y);
    return val;
}

// Function to perform warp-level sum reduction for a float3 value
float3 warpSum3(float3 val) {
    val.x = warpSum(val.x);
    val.y = warpSum(val.y);
    val.z = warpSum(val.z);
    return val;
}

// Main kernel for the backward pass
[numthreads(32, 32, 1)]
void ndRasterizeBackwardKernel(
    RasterizeBackwardParams params,          // Parameters for the kernel
    uint3 dispatchThreadID : SV_DispatchThreadID // Dispatch thread ID
) {
    uint i = dispatchThreadID.y; // Row index of the current thread
    uint j = dispatchThreadID.x; // Column index of the current thread
    float px = float(j) + 0.5;   // X-coordinate of the current pixel
    float py = float(i) + 0.5;   // Y-coordinate of the current pixel
    int pixId = min(i * int(params.imgSize.x) + j, int(params.imgSize.x * params.imgSize.y) - 1); // Pixel ID

    bool inside = (i < params.imgSize.y && j < params.imgSize.x); // Check if the current thread is inside the image bounds
    int2 range = params.tileBins[int(params.tileBounds.x * dispatchThreadID.z + dispatchThreadID.x)]; // Get the range of Gaussians for the current tile
    float vOutAlpha = params.vOutputAlpha[pixId]; // Output alpha value for the current pixel
    float TFinal = params.finalTs[pixId]; // Final T value for the current pixel
    float T = TFinal; // Initialize T with the final T value

    // Auto-diff storage for gradients
    @derivative float3 workspace[params.channels];

    // Initialize the workspace for each channel to zero
    for (uint c = 0; c < params.channels; ++c) {
        workspace[c] = 0.0f;
    }

    int binFinal = inside ? params.finalIndex[pixId] : 0; // Get the final bin index for the current pixel

    // Iterate over the Gaussians in the reverse order
    for (int idx = binFinal - 1; idx >= range.x; --idx) {
        bool valid = inside && idx < binFinal; // Check if the current Gaussian is valid
        int g = params.gaussiansIdsSorted[idx]; // Get the Gaussian ID
        float3 conic = params.conics[g]; // Get the conic coefficients for the Gaussian
        float2 center = params.xys[g]; // Get the center coordinates for the Gaussian
        float2 delta = center - float2(px, py); // Compute the difference between the center and the pixel coordinates
        float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y; // Compute the sigma value
        valid &= (sigma >= 0.0f); // Check if sigma is valid
        float opac = params.opacities[g]; // Get the opacity value for the Gaussian
        float vis = exp(-sigma); // Compute the visibility value
        float alpha = min(0.99f, opac * vis); // Compute the alpha value
        valid &= (alpha >= 1.0f / 255.0f); // Check if alpha is valid
        if (!any(valid)) {
            continue; // Skip the current Gaussian if it is not valid
        }
        @derivative float vAlpha = 0.0f; // Initialize the gradient with respect to alpha to zero
        @derivative float3 vConicLocal = float3(0.0f, 0.0f, 0.0f); // Initialize the local gradient with respect to conic coefficients to zero
        @derivative float2 vXYLocal = float2(0.0f, 0.0f); // Initialize the local gradient with respect to XY to zero
        @derivative float2 vXYAbsLocal = float2(0.0f, 0.0f); // Initialize the local absolute gradient with respect to XY to zero
        @derivative float vOpacityLocal = 0.0f; // Initialize the local gradient with respect to opacity to zero

        if (valid) {
            float ra = 1.0f / (1.0f - alpha); // Compute the reciprocal of (1 - alpha)
            T *= ra; // Update T
            float fac = alpha * T; // Compute the factor for the gradients
            for (uint c = 0; c < params.channels; ++c) {
                // Compute the gradient with respect to RGB values
                atomicAdd(params.vRgb[params.channels * g + c], fac * params.vOutput[params.channels * pixId + c]);
                vAlpha += (params.rgbs[params.channels * g + c] * T - workspace[c] * ra) * params.vOutput[params.channels * pixId + c];
                vAlpha += -TFinal * ra * params.background[c] * params.vOutput[params.channels * pixId + c];
                workspace[c] += params.rgbs[params.channels * g + c] * fac;
            }
            vAlpha += TFinal * ra * vOutAlpha; // Update the gradient with respect to alpha
            float vSigma = -opac * vis * vAlpha; // Compute the gradient with respect to sigma
            vConicLocal = float3(0.5f * vSigma * delta.x * delta.x, vSigma * delta.x * delta.y, 0.5f * vSigma * delta.y * delta.y); // Compute the local gradient with respect to conic coefficients
            vXYLocal = float2(vSigma * (conic.x * delta.x + conic.y * delta.y), vSigma * (conic.y * delta.x + conic.z * delta.y)); // Compute the local gradient with respect to XY
            vXYAbsLocal = abs(vXYLocal); // Compute the local absolute gradient with respect to XY
            vOpacityLocal = vis * vAlpha; // Compute the local gradient with respect to opacity
        }

        // Perform warp-level reductions using auto-diff
        vConicLocal = warpSum3(vConicLocal);
        vXYLocal = warpSum2(vXYLocal);
        vXYAbsLocal = warpSum2(vXYAbsLocal);
        vOpacityLocal = warpSum(vOpacityLocal);

        // Update the gradients with the reduced values
        if (dispatchThreadID.z % warpSize == 0) {
            atomicAdd(params.vConic[3 * g], vConicLocal.x);
            atomicAdd(params.vConic[3 * g + 1], vConicLocal.y);
            atomicAdd(params.vConic[3 * g + 2], vConicLocal.z);
            atomicAdd(params.vXY[2 * g], vXYLocal.x);
            atomicAdd(params.vXY[2 * g + 1], vXYLocal.y);
            atomicAdd(params.vXYAbs[2 * g], vXYAbsLocal.x);
            atomicAdd(params.vXYAbs[2 * g + 1], vXYAbsLocal.y);
            atomicAdd(params.vOpacity[g], vOpacityLocal);
        }
    }
}

