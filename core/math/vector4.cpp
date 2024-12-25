#include "vector4.h"
#include "core/string/ustring.h"  // For String class
#include "vector4i.h"            // For Vector4i class
#include <cstdio> // for snprintf

Vector4::operator String() const {
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "(%f, %f, %f, %f)", x, y, z, w);
    return String(buffer);
}

Vector4::operator Vector4i() const {
    return Vector4i(int(x), int(y), int(z), int(w));
}
