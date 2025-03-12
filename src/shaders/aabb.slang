public struct CameraUniform
{
    public float4x4 view;
    public float4x4 proj;
    public float4x4 view_proj;
};

public struct WorldUniform
{
    public float2 resolution;
};

struct VertexInput
{
    [vk_location(0)]
    float3 min;
    [vk_location(1)]
    float3 max;
    [vk_location(2)]
    float4 color;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float4 color;
};

[vk_binding(0, 0)]
ConstantBuffer<CameraUniform> camera;

[vk_binding(1, 0)]
ConstantBuffer<WorldUniform> world;

[shader("vertex")]
VertexOutput vertex_main(VertexInput vert, int vertexId: SV_VertexID)
{
    // Points for a 3D cube that are in order of the vertexId
    let points = float3[](
        float3(vert.min.x, vert.min.y, vert.min.z),
        float3(vert.max.x, vert.min.y, vert.min.z),
        float3(vert.max.x, vert.max.y, vert.min.z),
        float3(vert.min.x, vert.max.y, vert.min.z),
        float3(vert.min.x, vert.min.y, vert.min.z),
        //
        float3(vert.min.x, vert.min.y, vert.max.z),
        float3(vert.max.x, vert.min.y, vert.max.z),
        float3(vert.max.x, vert.min.y, vert.min.z),
        float3(vert.max.x, vert.min.y, vert.max.z),
        float3(vert.max.x, vert.max.y, vert.max.z),
        //
        float3(vert.max.x, vert.max.y, vert.min.z),
        float3(vert.max.x, vert.max.y, vert.max.z),
        float3(vert.min.x, vert.max.y, vert.max.z),
        float3(vert.min.x, vert.min.y, vert.max.z),
        float3(vert.min.x, vert.max.y, vert.max.z),
        float3(vert.min.x, vert.max.y, vert.min.z),
    );

    VertexOutput output;
    let pos = points[vertexId];

    output.position = mul(camera.view_proj, float4(pos, 1.0));
    output.color = vert.color;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input)
{
    return input.color;
}
