#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include"include\Util.hpp"
#include"include\Vector3.hpp"
#include"include\KDTree3.hpp"
using namespace std;
using namespace G3D;
using namespace Thea;
struct NamedPoint
{
    Vector3 position;
    size_t  id;

    NamedPoint() {}
  NamedPoint(float x, float y, float z)
    {
        position = Vector3(x, y, z);
    }
  NamedPoint(float x, float y, float z, const size_t _id)
    {
        id = _id;
        position = Vector3(x, y, z);
    }
};

namespace Thea {
    template <>
    struct PointTraits3<NamedPoint>
    {
        static Vector3 const & getPosition(NamedPoint const & np) { return np.position; }
    };

    template <>
    struct IsPoint3<NamedPoint>
    {
        static bool const value = true;
    };
}

typedef KDTree3<NamedPoint> PKDTree;