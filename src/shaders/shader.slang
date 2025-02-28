struct VertexOutput
{
    float4 position : SV_Position;
    [vk_location(0)]
    float3 vert_pos;
};

[shader("vertex")]
VertexOutput vertex_main(uint vertexId: SV_VertexID)
{
    VertexOutput output;
    output.position = float4(
        float(1 - int(vertexId)) * 0.5, float(int(vertexId & 1u) * 2 - 1) * 0.5, 0.0, 1.0);
    output.vert_pos = output.position.xyz;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input)
{
    return float4(1.0, 0.5, 0.0, 1.0);
}
