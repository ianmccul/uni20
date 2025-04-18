
TEST(Assign, Simple1D)
{
  std::vector<double> src_data = {1, 2, 3, 4};
  std::vector<double> dst_data(4, 0);

  auto src = make_mdspan_1d(src_data);
  auto dst = make_mdspan_1d(dst_data);

  uni20::assign(src, dst);

  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(dst_data[i], src_data[i]);
}

TEST(Assign, Strided2D)
{
  std::vector<double> buf1(25, 0.0), buf2(25, 0.0);

  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      buf1[r * 5 + c * 2] = static_cast<double>(r * 3 + c + 1);

  auto m1 = make_mdspan_2d(buf1, 3, 3, {5, 2});
  auto m2 = make_mdspan_2d(buf2, 3, 3, {5, 2});

  uni20::assign(m1, m2);

  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      EXPECT_EQ(buf2[r * 5 + c * 2], buf1[r * 5 + c * 2]);
}

TEST(Assign, Reversed1D)
{
  std::vector<double> v = {1, 2, 3, 4};
  std::vector<double> result(4, 0);

  auto src = make_reversed_1d(v);
  auto dst = make_mdspan_1d(result);

  uni20::assign(src, dst);

  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(result[i], v[3 - i]);
}

TEST(Assign, TransformNegate)
{
  std::vector<double> v = {1, 2, 3, 4};
  std::vector<double> out(4, 0);

  auto src = make_mdspan_1d(v);
  auto dst = make_mdspan_1d(out);

  auto neg = transform_view(src, [](double x) { return -x; });
  uni20::assign(neg, dst);

  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(out[i], -v[i]);
}

TEST(Assign, TransformScaleShift)
{
  std::vector<double> v = {1, 2, 3, 4};
  std::vector<double> out(4, 0);

  auto src = make_mdspan_1d(v);
  auto dst = make_mdspan_1d(out);

  auto chain = transform_view(transform_view(src, [](double x) { return 2 * x; }), [](double x) { return x + 1; });

  uni20::assign(chain, dst);

  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(out[i], 2 * v[i] + 1);
}
