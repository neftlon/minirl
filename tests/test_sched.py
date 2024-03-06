import pytest, math
from minirl.sched import LinearSchedule

@pytest.mark.parametrize("y1,y2,a,b", [(0., 1., 1., 0.), (1., 0., -1., 1.)])
def test_unit_cube(y1, y2, a, b):
  sched = LinearSchedule.bounded_within(0., 1., y1, y2)
  assert math.isclose(sched.a, a)
  assert math.isclose(sched.b, b)

@pytest.mark.parametrize("x1,x2", [(0.,5.), (3.,5.)])
@pytest.mark.parametrize("y1,y2", [(1.,4.), (4.,2.)])
def test_bounds(x1, x2, y1, y2):
  sched = LinearSchedule.bounded_within(x1, x2, y1, y2)
  assert sched(int(x1) - 1) == y1, "before x1, sched should be bounded"
  assert sched(int(x2) + 1) == y2, "after x2, sched should be bounded"

@pytest.mark.parametrize("x1,x2", [(0.,5.), (3.,5.)])
@pytest.mark.parametrize("y1,y2", [(1.,4.), (4.,2.)])
def test_close_at_extrema(x1, x2, y1, y2):
  sched = LinearSchedule.bounded_within(x1, x2, y1, y2)
  assert math.isclose(sched(int(x1)), y1), "at x1, sched should be first bound"
  assert math.isclose(sched(int(x2)), y2), "at x2, sched should be second bound"

if __name__ == "__main__":
  pytest.main([__file__])
