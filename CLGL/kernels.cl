__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void red(write_only image2d_t target)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	float4 red = (float4)(1.0f, 0.0f, 0.0f, 1.0f);

	if(x > 200 && x < 400 && y > 100 && y < 300){
		write_imagef(target, (int2) (x, y), red);
	}
}

__kernel void fill(write_only image2d_t target, float4 colour)
{
	//Make sure to get correct pixel coordinates: offset + work-item dims
	size_t x = get_global_offset(0) + get_global_id(0);
	size_t y = get_global_offset(1) + get_global_id(1);

	int2 pixel_pos = (int2)(x, y);

	write_imagef(target, pixel_pos, colour);
}

__kernel void bounding_box(__constant int2* in_verts, __global int4* out_rect)
{
	//ID of the triangle we're processing
	int id = get_global_id(0);
	
	//For indexing through the input array
	int in_id = id * 3;

	//Vertices:
	int2 v1 = in_verts[in_id];
	int2 v2 = in_verts[in_id + 1];
	int2 v3 = in_verts[in_id + 2];

	//Output: bounding rectangle data stored in a 4-vector
	//Format: (minX, minY, maxX, maxY)
	int4 bounds;

	//Comparisons
	//minx
	int temp = min(v1.x, v2.x);
	bounds.s0 = min(temp, v3.x);
	//miny
	temp = min(v1.y, v2.y);
	bounds.s1 = min(temp, v3.y);
	//maxx
	temp = max(v1.x, v2.x);
	bounds.s2 = max(temp, v3.x);
	//maxy
	temp = max(v1.y, v2.y);
	bounds.s3 = max(temp, v3.y);

	//Write to output
	out_rect[id] = bounds;
}

__kernel void half_space(__constant int2 *in_verts, __constant float4* in_colour, write_only image2d_t target)
{
	//Pixel coord
	int x = get_global_id(0);
	int y = get_global_id(1);

	//ID of triangle processed
	int tri_id = get_global_id(2);
	//Index for vertices array
	int index = tri_id * 3;

	//Vertices
	int2 v1 = in_verts[index];
	int2 v2 = in_verts[index + 1];
	int2 v3 = in_verts[index + 2];

	//Compute half-space functions
	int f1 = (v1.x - v2.x)*(y - v1.y) - (v1.y - v2.y)*(x - v1.x);
	int f2 = (v2.x - v3.x)*(y - v2.y) - (v2.y - v3.y)*(x - v2.x);
	int f3 = (v3.x - v1.x)*(y - v3.y) - (v3.y - v1.y)*(x - v3.x);

	//If all half-space functions are positive on this pixel, write the appropriate colour
	if(f1 > 0 && f2 > 0 && f3 > 0)
	{
		write_imagef(target, (int2)(x, y), in_colour[tri_id]);
	}
}

__kernel void half_space_box(__constant int2 *in_verts, __constant float4 *in_colour, write_only image2d_t target)
{
	//Pixel coord
	int x = get_global_id(0);
	int y = get_global_id(1);

	//Triangle ID
	int tri_id = get_global_id(2);
	//Index in vertex array
	int index = tri_id * 3;

	//Vertices
	int2 v1 = in_verts[index];
	int2 v2 = in_verts[index + 1];
	int2 v3 = in_verts[index + 2];

	//Check if pixel is inside triangle's bounding box
	//X-Coord
	int temp = min(v1.x, v2.x);
	int minm = min(temp, v3.x);
	temp = max(v1.x, v2.x);
	int maxm = max(temp, v3.x);

	if(x > minm && x < maxm)
	{
		//Y-Coord
		temp = min(v1.y, v2.y);
		minm = min(temp, v3.y);
		temp = max(v1.y, v2.y);
		maxm = max(temp, v3.y);

		if(y > minm && y < maxm)
		{
			//Pixel is inside bounding box
			//Compute half-space functions
			int f1 = (v1.x - v2.x)*(y - v1.y) - (v1.y - v2.y)*(x - v1.x);
			int f2 = (v2.x - v3.x)*(y - v2.y) - (v2.y - v3.y)*(x - v2.x);
			int f3 = (v3.x - v1.x)*(y - v3.y) - (v3.y - v1.y)*(x - v3.x);

			//If all half-space functions are positive on this pixel, write the appropriate colour
			if(f1 > 0 && f2 > 0 && f3 > 0)
			{
				write_imagef(target, (int2)(x, y), in_colour[tri_id]);
			}
		}
	}
}