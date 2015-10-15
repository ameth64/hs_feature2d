#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>

#include <gtest/gtest.h>

#include "hs_image_io/whole_io/image_io.hpp"
#include "hs_image_io/whole_io/image_data.hpp"

#include "hs_image_io/stream_io/stream_reader.hpp"

namespace
{

using namespace hs::imgio::whole;

class TestStreamReader
{
public:
  TestStreamReader(const std::string& path,
                   int width, int height, int channel, int bit_depth,
                   ImageData::ColorSpace color_type, int threshold = 0)
    : path_(path), width_(width), height_(height), channel_(channel),
      bit_depth_(bit_depth), color_type_(color_type), threshold_(threshold) {}

  enum TestResult
  {
    TR_SUCCESS = 0,
    TR_OPEN_IMAGE_ERROR,
    TR_READ_WRONG_TILE
  };

  TestResult Test()
  {
    hs::imgio::stream::StreamReader reader(path_);

    if (reader.width() != width_ || reader.height() != height_ ||
        reader.channel() != channel_ || reader.bit_depth() != bit_depth_ ||
        reader.color_type() != color_type_)
    {
      return TR_OPEN_IMAGE_ERROR;
    }

    int tile_width = reader.tile_width();
    int tile_height = reader.tile_height();

    uint64_t total_diff = 0;
    uint64_t number_of_tiles = 0;
    while (!reader.IsEndOfFile())
    {
      int tile_row_id, tile_col_id;
      ImageData image_data = reader.ReadNextTile(tile_row_id, tile_col_id);
      total_diff += TestTile(tile_width, tile_height,
                             tile_row_id, tile_col_id, image_data);
      number_of_tiles++;
    }
    total_diff /= number_of_tiles;
    if (total_diff > threshold_) return TR_READ_WRONG_TILE;

    //Second pass, Test Reset()
    reader.Reset();
    total_diff = 0;
    number_of_tiles = 0;
    while (!reader.IsEndOfFile())
    {
      int tile_row_id, tile_col_id;
      ImageData image_data = reader.ReadNextTile(tile_row_id, tile_col_id);
      total_diff += TestTile(tile_width, tile_height,
                             tile_row_id, tile_col_id, image_data);
      number_of_tiles++;
    }
    total_diff /= number_of_tiles;
    if (total_diff > threshold_) return TR_READ_WRONG_TILE;

    //Third pass, Test Open()
    if (reader.Open(path_) != 0 ||
        reader.width() != width_ || reader.height() != height_ ||
        reader.channel() != channel_ || reader.bit_depth() != bit_depth_ ||
        reader.color_type() != color_type_)
    {
      return TR_OPEN_IMAGE_ERROR;
    }
    total_diff = 0;
    number_of_tiles = 0;
    while (!reader.IsEndOfFile())
    {
      int tile_row_id, tile_col_id;
      ImageData image_data = reader.ReadNextTile(tile_row_id, tile_col_id);
      total_diff += TestTile(tile_width, tile_height,
                             tile_row_id, tile_col_id, image_data);
      number_of_tiles++;
    }
    total_diff /= number_of_tiles;
    if (total_diff > threshold_) return TR_READ_WRONG_TILE;

    return TR_SUCCESS;
  }

private:
  uint64_t TestTile(int tile_width, int tile_height,
               int tile_row_id, int tile_col_id,
               const ImageData& image_data) const
  {
    int tile_row_count = (height_ + tile_height - 1) / tile_height;
    int tile_col_count = (width_ + tile_width - 1) / tile_width;
    int actual_width = tile_col_id == tile_col_count - 1 ?
                       width_ - tile_col_id * tile_width : tile_width;
    int actual_height = tile_row_id == tile_row_count - 1 ?
                        height_ - tile_row_id * tile_height : tile_height;
    uint64_t number_of_samples = actual_height * actual_width * channel_;
    uint64_t sample_diff = 0;
    for (int i = 0; i < actual_height; i++)
    {
      for (int j = 0; j < actual_width; j++)
      {
        unsigned int x = (unsigned int)(tile_col_id * tile_width + j);
        unsigned int y = (unsigned int)(tile_row_id * tile_height + i);
        unsigned int pos = (unsigned int)(y << (channel_ * bit_depth_ / 2));
        pos += (unsigned int)(x);
        for (int k = 0; k < channel_; k++)
        {
          ImageData::Byte sample = image_data.GetByte(i, j, k);
          unsigned int mask = ((1 << bit_depth_) - 1) <<
                              ((channel_ - k - 1) * bit_depth_);
          ImageData::Byte expected_sample =
            ImageData::Byte((pos & mask) >> ((channel_ - k - 1) * bit_depth_));
          sample_diff += uint64_t(std::abs(sample - expected_sample));
        }
      }
    }

    sample_diff /= number_of_samples;
    return sample_diff;
  }
private:
  std::string path_;
  int width_;
  int height_;
  int channel_;
  int bit_depth_;
  ImageData::ColorSpace color_type_;
  int threshold_;
};


ImageData GenerateImageData(int width, int height, int channel, int bit_depth,
                            ImageData::ColorSpace color_type)
{
  ImageData image_data;
  image_data.CreateImage(width, height, channel, bit_depth, color_type);
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      unsigned int pos = (unsigned int)(i << (channel * bit_depth / 2));
      pos += (unsigned int)(j);
      for (int k = 0; k < channel; k++)
      {
        unsigned int mask = ((1 << bit_depth) - 1) <<
                            ((channel - k - 1) * bit_depth);
        ImageData::Byte sample =
          ImageData::Byte((pos & mask) >> ((channel - k - 1) * bit_depth));
        image_data.GetByte(i, j, k) = sample;
      }

    }
  }

  return image_data;
}


TEST(TestStreamReader, TIFTest)
{
  int width = 3967;
  int height = 3843;
  int channel = 3;
  int bit_depth = 8;
  std::string image_path = "tif_strip_channel_3_test.tif";
  ImageData::ColorSpace color_type = ImageData::IMAGE_RGB;
  ImageData image_data =
    GenerateImageData(width, height, channel, bit_depth, color_type);
  hs::imgio::whole::ImageIO().SaveTIFF(image_path, image_data, false);

  TestStreamReader tif_strip_channel_3_tester(
    image_path, width, height, channel, bit_depth, color_type);
  ASSERT_EQ(TestStreamReader::TR_SUCCESS, tif_strip_channel_3_tester.Test());

  width = 6000;
  height = 8000;
  channel = 4;
  bit_depth = 8;
  image_path = "tif_strip_channel_4_test.tif";
  color_type = ImageData::IMAGE_RGB;
  image_data =
    GenerateImageData(width, height, channel, bit_depth, color_type);
  hs::imgio::whole::ImageIO().SaveTIFF(image_path, image_data, false);

  TestStreamReader tif_strip_channel_4_tester(
    image_path, width, height, channel, bit_depth, color_type);
  ASSERT_EQ(TestStreamReader::TR_SUCCESS, tif_strip_channel_4_tester.Test());

  //TODO: width height must be times of 128,more general needed.
  width = 3968;
  height = 3840;
  channel = 3;
  bit_depth = 8;
  image_path = "tif_tile_channel_3_test.tif";
  color_type = ImageData::IMAGE_RGB;
  image_data =
    GenerateImageData(width, height, channel, bit_depth, color_type);
  hs::imgio::whole::ImageIO().SaveTIFF(image_path, image_data, true);

  TestStreamReader tif_tile_channel_3_tester(
    image_path, width, height, channel, bit_depth, color_type);
  ASSERT_EQ(TestStreamReader::TR_SUCCESS, tif_tile_channel_3_tester.Test());

  //TODO: none-3 channel support needed.
  //width = 4224;
  //height = 4352;
  //channel = 4;
  //bit_depth = 8;
  //image_path = "tif_tile_channel_4_test.tif";
  //color_type = ImageData::IMAGE_RGB;
  //image_data =
  //  GenerateImageData(width, height, channel, bit_depth, color_type);
  //hs::imgio::whole::ImageIO().SaveTIFF(image_path, image_data, true);

  //TestStreamReader tif_tile_channel_4_tester(
  //  image_path, width, height, channel, bit_depth, color_type);
  //ASSERT_EQ(TestStreamReader::TR_SUCCESS, tif_tile_channel_4_tester.Test());
}

TEST(StreamReader, JPGTest)
{
  int width = 3967;
  int height = 3843;
  int channel = 3;
  int bit_depth = 8;
  std::string image_path = "jpg_channel_3_test.jpg";
  ImageData::ColorSpace color_type = ImageData::IMAGE_RGB;
  ImageData image_data =
    GenerateImageData(width, height, channel, bit_depth, color_type);
  hs::imgio::whole::ImageIO().SaveJPEG(image_path, image_data, 95);

  TestStreamReader jpg_channel_3_tester(
    image_path, width, height, channel, bit_depth, color_type, 5);
  ASSERT_EQ(TestStreamReader::TR_SUCCESS, jpg_channel_3_tester.Test());

  width = 6000;
  height = 8000;
  channel = 4;
  bit_depth = 8;
  image_path = "jpg_channel_4_test.tif";
  color_type = ImageData::IMAGE_RGB;
  image_data =
    GenerateImageData(width, height, channel, bit_depth, color_type);
  hs::imgio::whole::ImageIO().SaveTIFF(image_path, image_data, false);

  TestStreamReader jpg_channel_4_tester(
    image_path, width, height, channel, bit_depth, color_type, 5);
  ASSERT_EQ(TestStreamReader::TR_SUCCESS, jpg_channel_4_tester.Test());
}

}
